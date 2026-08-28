"""Exercise the real running model with an official bundled image, never the user's camera."""
from io import BytesIO
from pathlib import Path
import importlib.util
import json
import httpx
from PIL import Image


def main():
    spec = importlib.util.find_spec('ultralytics')
    source = Path(spec.origin).parent / 'assets' / 'bus.jpg'
    with Image.open(source) as image:
        image.thumbnail((640, 480))
        buffer = BytesIO()
        image.convert('RGB').save(buffer, format='JPEG')
    # Do not use proxy environment variables for local frame transport.
    with httpx.Client(base_url='http://127.0.0.1:8001', trust_env=False, timeout=15) as client:
        health = client.get('/health'); health.raise_for_status()
        headers = {'X-Care-Client': 'active-care-web', 'Origin': 'http://localhost:3000'}
        session = client.post('/sessions', headers=headers); session.raise_for_status()
        headers['X-Session-Id'] = session.json()['session_id']
        try:
            response = client.post('/frames', content=buffer.getvalue(), headers={**headers, 'Content-Type':'image/jpeg', 'X-Frame-Id':'1'})
            response.raise_for_status()
            result = response.json()
            assert result['source'] == 'camera' and result['model'] == 'YOLO11n-pose'
            assert len(result['persons']) >= 1, 'Official bus fixture should contain detected people'
            assert all(len(p['keypoints']) == 17 for p in result['persons'])
            print(json.dumps({'fixture': 'Ultralytics bundled bus.jpg (not live camera)', 'persons': len(result['persons']), 'keypoints_per_person':17, 'inference_ms':result['inference_ms'], 'device':result['device']}, ensure_ascii=False))
        finally:
            client.delete('/sessions', headers=headers)


if __name__ == '__main__':
    main()
