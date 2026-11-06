"""Portable local launcher. Everything resolves relative to the extracted folder."""
import argparse
import atexit
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
import urllib.request
import webbrowser

BASE = Path(__file__).resolve().parent


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--port', type=int, default=None)
    args = parser.parse_args()
    os.chdir(BASE)
    data = BASE / 'user_data'
    logs = data / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    log_file = (logs/'startup.log').open('a',encoding='utf-8',buffering=1)
    original_out, original_err = sys.stdout, sys.stderr
    class Tee:
        def __init__(self, original): self.original = original
        def write(self, text):
            log_file.write(text)
            return self.original.write(text)
        def flush(self):
            log_file.flush()
            self.original.flush()
        def __getattr__(self, name): return getattr(self.original, name)
    sys.stdout, sys.stderr = Tee(original_out), Tee(original_err)
    def close_log():
        sys.stdout, sys.stderr = original_out, original_err
        log_file.close()
    atexit.register(close_log)
    captures = data / 'captures'
    captures.mkdir(exist_ok=True)
    os.environ['MINDGUARD_CAPTURES'] = str(captures)
    os.environ['MINDGUARD_SPEECH_PYTHON'] = sys.executable
    os.environ['MINDGUARD_MODEL_STARTUP_TIMEOUT'] = '600'
    os.environ['HF_HOME'] = str(data / 'cache')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['PYTHONUTF8'] = '1'
    sys.path.insert(0, str(BASE / 'app'))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    state_file = logs / 'running.json'
    # Reopen only a service belonging to THIS portable installation.
    if args.port is None and state_file.exists():
        try:
            previous = json.loads(state_file.read_text(encoding='utf-8'))
            port = int(previous['port'])
            if not 1024 <= port <= 65535:
                raise ValueError('Invalid port')
            url = f'http://127.0.0.1:{port}'
            with opener.open(url+'/api/status', timeout=2) as response:
                status = json.load(response)
            if Path(status['captures_directory']).resolve() == captures.resolve():
                print('Already running: ' + url, flush=True)
                if not args.no_browser:
                    webbrowser.open(url)
                return
        except Exception:
            pass
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    for port in ([args.port] if args.port is not None else range(8765, 8796)):
        try:
            listener.bind(('127.0.0.1', port))
            break
        except OSError:
            if args.port is not None or port == 8795:
                raise RuntimeError('No available local port (8765-8795).')
    listener.listen(128)
    actual_port = listener.getsockname()[1]
    url = f'http://127.0.0.1:{actual_port}'
    state_file.write_text(json.dumps({'port':actual_port,'pid':os.getpid(),'url':url}), encoding='utf-8')
    print('\n御郁先锋 / Vanguard\n正在加载三个真实模型，首次启动可能需要数分钟。\n'
          '请保留此窗口；网页会在模型加载后自动打开。\n'
          f'本机地址：{url}\n停止服务：按 Ctrl+C。\n', flush=True)
    stop = threading.Event()

    def open_when_ready():
        while not stop.wait(1):
            try:
                with opener.open(url+'/api/status', timeout=2) as response:
                    result = json.load(response)
                ready = sum(model['status']=='ready' for model in result['models'].values())
                print(f'模型加载：{ready}/3。打开 {url}', flush=True)
                for name, model in result['models'].items():
                    if model['status'] != 'ready':
                        print(name + ': ' + model.get('error','unknown error'), flush=True)
                if not args.no_browser:
                    webbrowser.open(url)
                return
            except Exception:
                continue

    threading.Thread(target=open_when_ready, daemon=True).start()
    try:
        import uvicorn
        from trimodal.server import app
        config = uvicorn.Config(app, host='127.0.0.1', port=actual_port, log_level='info')
        uvicorn.Server(config).run(sockets=[listener])
    finally:
        stop.set()
        listener.close()
        try:
            if json.loads(state_file.read_text(encoding='utf-8'))['pid'] == os.getpid():
                state_file.unlink()
        except (OSError, ValueError, KeyError):
            pass


if __name__ == '__main__':
    main()
