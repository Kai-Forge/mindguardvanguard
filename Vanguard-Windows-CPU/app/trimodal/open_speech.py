"""Official emotion2vec+ base adapter with explicit nine-to-three mapping."""
import atexit
import base64
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import numpy as np

HERE=Path(__file__).resolve().parent
RAW_CLASSES=('angry','disgusted','fearful','happy','neutral','other','sad','surprised','unknown')

def map_emotions(labels,scores):
    canonical=[label.split('/')[-1] if label!='<unk>' else 'unknown' for label in labels]
    if len(canonical)!=9 or set(canonical)!=set(RAW_CLASSES):
        raise ValueError(f'Unexpected emotion2vec class labels: {labels}')
    values=np.asarray(scores,dtype=np.float64)
    if values.shape!=(9,) or not np.isfinite(values).all() or (values<0).any() or not np.isclose(values.sum(),1,atol=1e-5):
        raise ValueError('Invalid emotion2vec probabilities')
    mapping=dict(zip(canonical,values))
    p=np.array([sum(mapping[k] for k in ('angry','disgusted','fearful','sad')),mapping['neutral'],mapping['happy']])
    retained=float(p.sum())
    if retained<1e-6:
        raise ValueError('语音主要为惊讶/其他/未知，无法可靠映射三分类')
    excluded={key:float(mapping[key]) for key in ('other','surprised','unknown')}
    return p/retained,dict(original_classes=canonical,original_probabilities=values.tolist(),
                           excluded_probabilities=excluded,excluded_probability=sum(excluded.values()),
                           mapping='negative=angry+disgusted+fearful+sad; neutral=neutral; positive=happy; exclude other/surprised/unknown')

class OpenSpeechClassifier:
    sample_rate=16000
    name='emotion2vec+ base · 官方预训练'
    def __init__(self):
        self.path=HERE/'assets/emotion2vec_plus_base/model.pt'
        python=Path(os.environ.get('MINDGUARD_SPEECH_PYTHON', str(HERE/'.venv_ser/Scripts/python.exe')))
        if not self.path.is_file() or not python.is_file():
            raise FileNotFoundError('emotion2vec weights/runtime missing; follow OPEN_SPEECH.md')
        (HERE/'outputs').mkdir(exist_ok=True)
        self.log=(HERE/'outputs/emotion2vec_worker.log').open('a',encoding='utf-8')
        self.process=subprocess.Popen([str(python),'-X','utf8','-u',str(HERE/'speech_worker.py')],
                                      stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,
                                      text=True,encoding='utf-8',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.responses=queue.Queue()
        self.lock=threading.Lock()
        def reader():
            try:
                for line in self.process.stdout:
                    self.responses.put(json.loads(line))
            except Exception as exc:
                self.responses.put(dict(status='error',error=str(exc)))
            finally:
                self.responses.put(dict(status='error',error='emotion2vec worker exited'))
        threading.Thread(target=reader,daemon=True).start()
        try:
            self.runtime=self._receive(float(os.environ.get('MINDGUARD_MODEL_STARTUP_TIMEOUT', '180')))
            if self.runtime['status']!='ready':
                raise RuntimeError(self.runtime.get('error','Model startup failed'))
        except Exception:
            self.close();raise
        atexit.register(self.close)

    def _receive(self,timeout):
        try:
            response=self.responses.get(timeout=timeout)
        except queue.Empty:
            self.close()
            raise TimeoutError('emotion2vec inference timed out; restart service to reload worker')
        if response['status']=='error':
            raise RuntimeError(response['error'])
        return response

    def predict(self,waveform):
        samples=np.asarray(waveform,dtype='<f4')
        if samples.ndim!=1 or not 800<=len(samples)<=480000 or not np.isfinite(samples).all():
            raise ValueError('Expected finite 16 kHz mono waveform, 0.05-30 seconds')
        with self.lock:
            if self.process.poll() is not None:
                raise RuntimeError('emotion2vec worker is not running')
            request=dict(float32_base64=base64.b64encode(samples.tobytes()).decode('ascii'))
            self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush()
            result=self._receive(60)
        p,details=map_emotions(result['labels'],result['scores'])
        details.update(inference_ms=result['inference_ms'],device=result['device'],input_sample_rate=16000)
        return p,details

    def close(self):
        if getattr(self,'process',None) is not None and self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:self.process.kill()
        if getattr(self,'log',None) is not None and not self.log.closed:self.log.close()
