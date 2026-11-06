"""Persistent isolated FunASR runtime; stdout contains JSON responses only."""
import base64
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import time

def main():
    pipe=sys.stdout
    def send(value):
        pipe.write(json.dumps(value,ensure_ascii=True,allow_nan=False)+'\n');pipe.flush()
    try:
        with redirect_stdout(sys.stderr):
            import numpy as np
            import torch
            from funasr import AutoModel
            torch.set_num_threads(2)
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            folder=Path(__file__).parent/'assets/emotion2vec_plus_base'
            device='cuda:0' if torch.cuda.is_available() else 'cpu'
            model=AutoModel(model=str(folder),device=device,disable_update=True,
                            disable_pbar=True,log_level='ERROR')
            model.model.eval()
        send(dict(status='ready',device=device,parameters=sum(p.numel() for p in model.model.parameters())))
    except Exception as exc:
        import traceback
        traceback.print_exc(file=sys.stderr)
        send(dict(status='error',error=str(exc)));return
    for line in sys.stdin:
        try:
            data=json.loads(line)
            samples=np.frombuffer(base64.b64decode(data['float32_base64']),dtype='<f4').copy()
            if not 800<=len(samples)<=480000 or not np.isfinite(samples).all():
                raise ValueError('Expected 0.05-30 seconds of finite 16 kHz mono samples')
            start=time.perf_counter()
            with redirect_stdout(sys.stderr),torch.inference_mode():
                output=model.generate(input=samples,fs=16000,granularity='utterance',
                                      extract_embedding=False,disable_pbar=True)[0]
            send(dict(status='ok',labels=output['labels'],scores=output['scores'],
                      inference_ms=(time.perf_counter()-start)*1000,device=device))
        except Exception as exc:
            send(dict(status='error',error=str(exc)))

if __name__=='__main__':main()
