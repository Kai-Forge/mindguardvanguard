"""python -m trimodal.infer --image FACE.jpg --wav S70.wav --demo-unpaired"""
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path
import cv2
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from mindguard.features import make_mel_transform, waveforms_to_logmel_variable
from mindguard.model import SERModel
from .fusion import CLASSES, FACE_CLASSES, softmax, face_to_valence, fuse

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class SignalClassifier:
    sample_rate = 8000
    def __init__(self, checkpoint, modality):
        self.path = Path(checkpoint)
        cp = torch.load(self.path, map_location='cpu', weights_only=True)
        self.config = cp['config']
        if tuple(cp['classes']) != CLASSES or self.config['modality'] != modality or cp['input_channels'] != 1:
            raise ValueError(f'Checkpoint class/modality mismatch: {checkpoint}')
        if not self.config['masked_feature_padding']:
            raise ValueError('Padding-safe checkpoint required')
        self.model = SERModel(input_channels=1, normalization=self.config['normalization']).eval()
        self.model.load_state_dict(cp['model'], strict=True)
        self.mel = make_mel_transform()

    @torch.inference_mode()
    def predict(self, waveform):
        x = torch.as_tensor(np.ascontiguousarray(waveform), dtype=torch.float32)[None, None, :]
        if x.shape[-1] <= 200 or not torch.isfinite(x).all():
            raise ValueError('Signal must be finite and longer than 25 ms')
        features, lengths = waveforms_to_logmel_variable(
            x, torch.tensor([x.shape[-1]]), self.mel,
            power_db_multiplier=float(self.config['power_db_multiplier']))
        if lengths.item() < 4:
            raise ValueError('Signal too short for CNN pooling')
        logits = self.model(features, lengths, mask_cnn_padding=True)[0].numpy()
        return softmax(logits), logits.tolist()

class ImageClassifier:
    def __init__(self, detector, expression):
        # Original upstream alignment/preprocessing, preserved with its notice.
        file = Path(__file__).parent / 'assets' / 'facial_fer_model.py'
        spec = importlib.util.spec_from_file_location('opencv_zoo_fer', file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.fer = module.FacialExpressionRecog(str(expression))
        self.detector = cv2.FaceDetectorYN.create('onnx', np.fromfile(detector, dtype=np.uint8),
                                                np.empty(0, dtype=np.uint8), (320, 320), 0.8, 0.3, 5000)

    def predict(self, path, face_index=None):
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Cannot decode image: {path}')
        self.detector.setInputSize((image.shape[1], image.shape[0]))
        _, faces = self.detector.detect(image)
        if faces is None or len(faces) == 0:
            raise ValueError('No face detected; provide a clear face image')
        faces = sorted(faces, key=lambda row: (float(row[0]), float(row[1])))
        if face_index is None and len(faces) != 1:
            raise ValueError(f'{len(faces)} faces detected; select --face-index (left-to-right, zero-based)')
        index = 0 if face_index is None else face_index
        if not 0 <= index < len(faces):
            raise ValueError('face-index out of range')
        face = faces[index]
        blob = self.fer._preprocess(image, face[:-1])
        self.fer._model.setInput(blob, self.fer._inputNames)
        logits = self.fer._model.forward(self.fer._outputNames)[0].reshape(-1)
        if logits.shape != (7,):
            raise ValueError('Unexpected facial model output shape')
        p7 = softmax(logits)
        p3, surprise = face_to_valence(p7)
        return p3, dict(original_classes=list(FACE_CLASSES), logits=logits.tolist(),
                        original_probabilities=p7.tolist(), excluded_surprise_probability=surprise,
                        mapping='negative=angry+disgust+fearful+sad; neutral=neutral; positive=happy; condition on non-surprise',
                        face_count=len(faces), face_index=index, bbox=face[:4].tolist())

def load_signals(path, target_sr=8000):
    samples, sr = sf.read(path, dtype='float32', always_2d=True)
    if samples.shape[1] != 2 or len(samples) == 0 or not np.isfinite(samples).all():
        raise ValueError('Expected nonempty finite stereo WAV: channel 0 speech, channel 1 EGG')
    if target_sr is not None and sr != target_sr:
        samples = AF.resample(torch.from_numpy(samples.T.copy()), sr, target_sr).T.numpy()
    return samples, sr

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--wav', type=Path, required=True, help='Stereo speech+EGG recording for one analysis window')
    parser.add_argument('--face-index', type=int)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'config.json')
    parser.add_argument('--weights', nargs=3, type=float, metavar=('IMAGE', 'SPEECH', 'EGG'))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--paired', action='store_true', help='Caller declares same subject/window; no biometric verification')
    mode.add_argument('--demo-unpaired', action='store_true', help='Engineering demonstration only, not a real fused emotion observation')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if tuple(config['classes']) != CLASSES or config['surprise_policy'] != 'exclude_and_renormalize':
        raise ValueError('Unsupported label mapping')
    weights = config['weights'] if args.weights is None else dict(zip(('image', 'speech', 'egg'), args.weights))
    fuse({n: [1/3]*3 for n in weights}, weights)  # validate before loading models
    started = time.perf_counter()
    samples, original_sr = load_signals(args.wav, target_sr=None)
    probs, details = {}, {}
    for name, channel in [('speech', 0), ('egg', 1)]:
        if name == 'speech' and config.get('speech_backend') == 'emotion2vec_plus_base':
            from .open_speech import OpenSpeechClassifier
            model = OpenSpeechClassifier()
            checkpoint = model.path
        else:
            checkpoint = ROOT / config[name + '_checkpoint']
            model = SignalClassifier(checkpoint, name)
        signal = torch.from_numpy(samples[:,channel].copy())
        if original_sr != model.sample_rate:
            signal = AF.resample(signal, original_sr, model.sample_rate)
        probs[name], information = model.predict(signal.numpy())
        details[name] = dict(checkpoint=str(checkpoint), sha256=digest(checkpoint), input_sample_rate=model.sample_rate)
        details[name]['details' if isinstance(information,dict) else 'logits'] = information
        if hasattr(model, 'close'):
            model.close()
    face = ImageClassifier(ROOT / config['face_detector'], ROOT / config['expression_model'])
    probs['image'], details['image'] = face.predict(args.image, args.face_index)
    for key in ['face_detector', 'expression_model']:
        details['image'][key + '_sha256'] = digest(ROOT / config[key])
    final = fuse(probs, weights)
    result = dict(status='COMPLETED', mode='caller_declared_paired' if args.paired else 'UNPAIRED_ENGINEERING_DEMO',
                  pairing_verified=False, scientific_evidence=False, classes=list(CLASSES), weights=weights,
                  probabilities={k: v.tolist() for k, v in probs.items()}, fused_probabilities=final.tolist(),
                  prediction=CLASSES[int(final.argmax())], model_details=details,
                  inputs=dict(image=str(args.image.resolve()), wav=str(args.wav.resolve()),
                              image_sha256=digest(args.image), wav_sha256=digest(args.wav),
                              original_sample_rate=original_sr, duration_seconds=len(samples)/original_sr),
                  elapsed_seconds=time.perf_counter()-started,
                  note='Probability normalization is not confidence calibration; surprise excluded explicitly.')
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding='utf-8')
    print(output)

if __name__ == '__main__':
    main()
