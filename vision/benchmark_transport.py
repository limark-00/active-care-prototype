"""Local HTTP cadence benchmark using an official fixture, NOT camera latency/FPS."""
import argparse
from io import BytesIO
import importlib.util
import json
from pathlib import Path
import statistics
import time

import httpx
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=20)
    parser.add_argument('--interval-ms', type=float, default=300)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 5 <= args.frames <= 100 or not 0 <= args.interval_ms <= 1000:
        parser.error('frames must be 5–100; interval-ms must be 0–1000')
    spec = importlib.util.find_spec('ultralytics')
    with Image.open(Path(spec.origin).parent / 'assets' / 'bus.jpg') as source:
        source.thumbnail((640, 480))
        data = BytesIO()
        source.convert('RGB').save(data, format='JPEG', quality=80)
    headers = {'X-Care-Client': 'active-care-web', 'Origin': 'http://localhost:3000'}
    roundtrips, inference, completed, counts = [], [], [], []
    with httpx.Client(base_url='http://127.0.0.1:8001', trust_env=False, timeout=15) as client:
        health = client.get('/health'); health.raise_for_status()
        session = client.post('/sessions', headers=headers); session.raise_for_status()
        headers['X-Session-Id'] = session.json()['session_id']
        try:
            for index in range(args.frames + 2):
                started = time.perf_counter()
                response = client.post('/frames', content=data.getvalue(), headers={**headers, 'Content-Type': 'image/jpeg', 'X-Frame-Id': str(index)})
                response.raise_for_status()
                received = time.perf_counter()
                result = response.json()
                if index >= 2:  # Exclude connection/model warm-up.
                    roundtrips.append((received - started) * 1000)
                    inference.append(result['inference_ms'])
                    completed.append(received)
                    counts.append(len(result['persons']))
                time.sleep(max(0, args.interval_ms / 1000 - (time.perf_counter() - started)))
        finally:
            client.delete('/sessions', headers=headers)
    report = {
        'scope': 'Official bus.jpg via local HTTP; synthetic pacing. NOT browser/live camera FPS or accuracy.',
        'frames': args.frames, 'interval_ms': args.interval_ms, 'device': health.json()['device'],
        'http_median_ms': round(statistics.median(roundtrips), 1),
        'http_p95_ms': round(sorted(roundtrips)[int((len(roundtrips)-1)*0.95)], 1),
        'inference_median_ms': round(statistics.median(inference), 1),
        'completed_fps': round((len(completed)-1)/(completed[-1]-completed[0]), 2),
        'detected_people_min_max': [min(counts), max(counts)],
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(output + '\n')
    print(output)


if __name__ == '__main__':
    main()
