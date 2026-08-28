"""Explicit setup download from the Ultralytics official GitHub release."""
import hashlib
from pathlib import Path
from urllib.request import urlopen

URL = 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-pose.pt'
ROOT = Path(__file__).resolve().parent


def main():
    destination = ROOT / 'models' / 'yolo11n-pose.pt'
    destination.parent.mkdir(exist_ok=True)
    expected_path = ROOT / 'model.sha256'
    expected = expected_path.read_text().split()[0] if expected_path.exists() else None
    if not destination.exists():
        with urlopen(URL, timeout=120) as response:
            data = response.read(20_000_000)
        if len(data) < 1_000_000 or len(data) >= 20_000_000:
            raise RuntimeError('模型大小异常，停止保存。')
        if expected and hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError('模型 SHA256 不匹配，停止保存。')
        destination.write_bytes(data)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    if expected and digest != expected:
        raise RuntimeError('已有模型 SHA256 不匹配，请检查文件来源。')
    print(f'{digest}  {destination.name}')


if __name__ == '__main__':
    main()
