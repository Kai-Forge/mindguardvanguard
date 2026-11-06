"""Local UI backed by the three real trimodal checkpoints. No synthetic scores."""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import uuid
import zipfile

import cv2
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from .infer import ROOT, SignalClassifier, ImageClassifier, digest
from .fusion import CLASSES, fuse
from .open_speech import OpenSpeechClassifier

HERE = Path(__file__).resolve().parent
CAPTURES = Path(os.environ.get('MINDGUARD_CAPTURES', str(HERE / 'captures'))).resolve()
LOCK = threading.RLock()
MODELS, MODEL_INFO = {}, {}

def utc():
    return datetime.now(timezone.utc).isoformat()

def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)

def load_models():
    for model in MODELS.values():
        if hasattr(model, 'close'):
            model.close()
    MODELS.clear()
    MODEL_INFO.clear()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
    for name in ('speech', 'egg', 'image'):
        try:
            if name == 'speech' and config.get('speech_backend') == 'emotion2vec_plus_base':
                model = OpenSpeechClassifier()
                path = model.path
                title = model.name
            elif name == 'image':
                path = ROOT / config['expression_model']
                model = ImageClassifier(ROOT / config['face_detector'], path)
                title = 'MobileFaceNet · OpenCV Zoo'
            else:
                path = ROOT / config[name + '_checkpoint']
                model = SignalClassifier(path, name)
                title = f'{name.upper()}-only · CNN–BiLSTM · seed 42'
            MODELS[name] = model
            MODEL_INFO[name] = dict(status='ready', name=title, checkpoint=str(path), sha256=digest(path))
            if name == 'speech':
                MODEL_INFO[name]['input_sample_rate'] = getattr(model, 'sample_rate', 8000)
                MODEL_INFO[name]['runtime'] = getattr(model, 'runtime', {'device':'cpu'})
        except Exception as exc:
            MODEL_INFO[name] = dict(status='error', name=name, error=str(exc))

@asynccontextmanager
async def lifespan(app):
    CAPTURES.mkdir(parents=True, exist_ok=True)
    load_models()
    yield
    for model in MODELS.values():
        if hasattr(model, 'close'):
            model.close()

app = FastAPI(lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', 'testserver'])

@app.middleware('http')
async def local_only(request, call_next):
    origin = request.headers.get('origin')
    if origin and origin != str(request.base_url).rstrip('/'):
        return Response('Cross-origin requests are not allowed', status_code=403)
    response = await call_next(request)
    response.headers['Cache-Control'] = 'no-store'
    return response

@app.get('/')
def home():
    return FileResponse(HERE / 'web' / 'index.html')

@app.get('/api/status')
def status():
    return dict(models=MODEL_INFO, classes=CLASSES, egg_live_connected=False,
                captures_directory=str(CAPTURES), time=utc())

class SessionRequest(BaseModel):
    participant: str = Field(min_length=1, max_length=80)
    label: str = 'unlabeled'
    notes: str = Field(default='', max_length=2000)

def session_dir(identifier):
    try:
        if str(uuid.UUID(identifier)) != identifier:
            raise ValueError()
    except ValueError:
        raise HTTPException(400, 'Invalid session ID')
    path = CAPTURES / identifier
    if not (path / 'session.json').is_file():
        raise HTTPException(404, 'Session not found')
    return path

@app.post('/api/sessions')
def create_session(body: SessionRequest):
    if body.label not in (*CLASSES, 'unlabeled'):
        raise HTTPException(400, 'Invalid human label')
    with LOCK:
        identifier = str(uuid.uuid4())
        directory = CAPTURES / identifier
        directory.mkdir(parents=True)
        record = dict(id=identifier, participant=body.participant.strip(), human_label=body.label,
                      notes=body.notes, created_at=utc(), status='recording', sample_count=0,
                      egg_live_connected=False, models=MODEL_INFO,
                      capture_format='Non-overlapping audio windows with representative JPEG frames; not continuous video',
                      sync='Browser clock timestamps; software association, not hardware synchronization')
        write_json(directory / 'session.json', record)
    return record

@app.post('/api/sessions/{identifier}/stop')
def stop_session(identifier: str):
    with LOCK:
        directory = session_dir(identifier)
        record = json.loads((directory / 'session.json').read_text(encoding='utf-8'))
        record.update(status='stopped', stopped_at=utc())
        write_json(directory / 'session.json', record)
    return record

@app.get('/api/sessions')
def list_sessions():
    with LOCK:
        return sorted([json.loads(p.read_text(encoding='utf-8')) for p in CAPTURES.glob('*/session.json')],
                      key=lambda item: item['created_at'], reverse=True)[:100]

@app.get('/api/sessions/{identifier}/download')
def download(identifier: str):
    with LOCK:
        directory = session_dir(identifier)
        memory = io.BytesIO()
        with zipfile.ZipFile(memory, 'w', zipfile.ZIP_DEFLATED) as archive:
            for file in directory.rglob('*'):
                if file.is_file() and file.suffix != '.tmp':
                    archive.write(file, file.relative_to(directory))
    return Response(memory.getvalue(), media_type='application/zip',
                    headers={'Content-Disposition': f'attachment; filename="mindguard-{identifier}.zip"'})

def absent(message):
    return dict(status='unavailable', message=message, probabilities=None, prediction=None)

def prediction(p, **extra):
    return dict(status='ok', probabilities=p.tolist(), prediction=CLASSES[int(p.argmax())], **extra)

def read_upload(upload, maximum):
    if upload is None:
        return None
    data = upload.file.read(maximum + 1)
    if not data or len(data) > maximum:
        raise HTTPException(400, 'Uploaded file empty or too large')
    return data

@app.post('/api/analyze')
def analyze(image: UploadFile | None = File(None), audio: UploadFile | None = File(None),
            mode: str = Form('live'), session_id: str = Form(''),
            timing: str = Form('{}'), weights: str = Form('[1,1,1]'), face_index: int = Form(-1)):
    if mode not in ('live', 'file'):
        raise HTTPException(400, 'Invalid input mode')
    try:
        stamps = json.loads(timing)
        values = json.loads(weights)
        if not isinstance(stamps, dict) or len(timing) > 4000 or not isinstance(values, list) or len(values) != 3:
            raise ValueError('Invalid metadata')
        weight_map = dict(zip(('image','speech','egg'), values))
        fuse({name: [1/3]*3 for name in weight_map}, weight_map)
        json.dumps(stamps, allow_nan=False)
    except (ValueError, TypeError):
        raise HTTPException(400, 'Invalid weights or timing metadata')
    image_data = read_upload(image, 8 * 1024 * 1024)
    audio_data = read_upload(audio, 20 * 1024 * 1024)
    if not image_data and not audio_data:
        raise HTTPException(400, 'No input')
    with LOCK:  # OpenCV networks and session writes are serialized.
        directory = None
        if session_id:
            if mode != 'live':
                raise HTTPException(400, 'File demonstrations cannot enter a live collection session')
            directory = session_dir(session_id)
            session = json.loads((directory / 'session.json').read_text(encoding='utf-8'))
            if session['status'] != 'recording':
                raise HTTPException(409, 'Session has stopped')
        began = time.perf_counter()
        result = dict(id=str(uuid.uuid4()), source=mode, received_at=utc(), client_timing=stamps,
                      classes=CLASSES, models=MODEL_INFO, weights=weight_map,
                      image=absent('未提供图像'), speech=absent('未提供语音'),
                      egg=absent('无 EGG 接入，无推理结果'),
                      fusion=absent('等待三路有效输入'), saved=False)
        # Validate and preserve original inputs; no fabricated replacement on failure.
        try:
            if image_data:
                decoded = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
                if decoded is None or decoded.shape[0] * decoded.shape[1] > 20000000:
                    raise ValueError('图像无法解码或尺寸过大')
                if 'image' not in MODELS:
                    raise ValueError('图像模型未加载：' + MODEL_INFO['image'].get('error',''))
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / 'frame.jpg'
                    path.write_bytes(image_data)
                    p, details = MODELS['image'].predict(path, None if face_index < 0 else face_index)
                result['image'] = prediction(p, details=details)
        except Exception as exc:
            result['image'] = dict(status='error', message=str(exc), probabilities=None, prediction=None)
        if audio_data:
            try:
                with sf.SoundFile(io.BytesIO(audio_data)) as file:
                    sr, channels, duration = file.samplerate, file.channels, len(file)/file.samplerate
                    if sr < 4000 or sr > 192000 or not .05 <= duration <= 30 or channels not in (1,2):
                        raise ValueError('需要 0.05–30 秒、单/双通道音频，采样率4–192kHz')
                    if mode == 'live' and channels != 1:
                        raise ValueError('实时麦克风仅接受单声道，不把第二声道解释为 EGG')
                    samples = file.read(dtype='float32', always_2d=True)
                if not np.isfinite(samples).all():
                    raise ValueError('音频包含非有限数值')
                result['audio_info'] = dict(sample_rate=sr, channels=channels, seconds=duration,
                                          mapping='ch0=speech; ch1=EGG only for stereo file mode')
                for name, index in [('speech',0)] + ([('egg',1)] if mode == 'file' and channels == 2 else []):
                    try:
                        channel = samples[:,index]
                        if float(np.std(channel)) < 1e-7:
                            result[name] = absent('信号为空或近似常量，无推理结果')
                            continue
                        if name not in MODELS:
                            raise ValueError('模型未加载：' + MODEL_INFO[name].get('error',''))
                        target_sr = getattr(MODELS[name], 'sample_rate', 8000)
                        if sr != target_sr:
                            channel = AF.resample(torch.from_numpy(channel.copy()), sr, target_sr).numpy()
                        p, information = MODELS[name].predict(channel)
                        if isinstance(information, dict):
                            result[name] = prediction(p, details=information)
                        else:
                            result[name] = prediction(p, logits=information)
                        result[name]['input_sample_rate'] = target_sr
                    except Exception as exc:
                        result[name] = dict(status='error', message=str(exc), probabilities=None, prediction=None)
            except Exception as exc:
                result['speech'] = dict(status='error', message=str(exc), probabilities=None, prediction=None)
        if all(result[name]['status'] == 'ok' for name in ('image','speech','egg')):
            p = fuse({name: result[name]['probabilities'] for name in weight_map}, weight_map)
            result['fusion'] = prediction(p, message='文件输入加权结果；是否同人同时间由输入者核实')
        result['elapsed_ms'] = round((time.perf_counter()-began)*1000, 1)
        if directory:
            sample_dir = directory / result['id']
            sample_dir.mkdir()
            if image_data:
                (sample_dir/'frame.jpg').write_bytes(image_data)
            if audio_data:
                (sample_dir/'audio.wav').write_bytes(audio_data)
            result.update(saved=True, session_id=session_id, human_label=session['human_label'],
                          participant=session['participant'])
            write_json(sample_dir/'prediction.json', result)
            session['sample_count'] += 1
            session['last_received_at'] = result['received_at']
            write_json(directory/'session.json', session)
        return result

app.mount('/static', StaticFiles(directory=HERE/'web'), name='static')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8765)
