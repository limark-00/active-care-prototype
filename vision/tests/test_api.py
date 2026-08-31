from io import BytesIO
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from app import create_app


class FakeDetector:
    model_name = 'test-double'
    device = 'cpu'
    def predict(self, image):
        return []


class FakeDecisionModel:
    model_name = 'decision-test-double'
    device = 'cpu'
    run_name = 'partial-macbert-v31-test'
    created = 0
    def __init__(self):
        type(self).created += 1
    def predict(self, text):
        def prediction(label, values):
            probability = round(1 / len(values), 6)
            probabilities = {str(value).lower() if isinstance(value, bool) else value: probability for value in values}
            probabilities[str(label).lower() if isinstance(label, bool) else label] += 1 - sum(probabilities.values())
            return {'label': label, 'confidence': probabilities[str(label).lower() if isinstance(label, bool) else label], 'probabilities': probabilities}
        return {
            'schema_version': 1,
            'source': 'local_text_model',
            'model': self.model_name,
            'base_model': 'test-base',
            'run_name': self.run_name,
            'device': self.device,
            'input_characters': len(text),
            'segment_count': 2,
            'inference_ms': 12.3,
            'model_output': {
                'risk_level': prediction('L2', ['L0','L1','L2','L3','L4']),
                'intervention_level': prediction('I2', ['I0','I1','I2','I3','I4']),
                'alert_mode': prediction('PAGE_WARNING', ['NONE','PAGE_WARNING','URGENT_HELP']),
                'manual_review': prediction(False, [False, True]),
                'abstain': prediction(False, [False, True]),
            },
            'guarded_output': {'risk_level':'L2','intervention_level':'I2','alert_mode':'PAGE_WARNING','manual_review':False,'abstain':False},
            'guardrail_applied': False,
            'review_reasons': [],
            'limitations': 'test',
        }


@pytest.fixture
def client():
    FakeDecisionModel.created = 0
    with TestClient(create_app(FakeDetector, FakeDecisionModel)) as client:
        yield client


HEADERS = {'Origin':'http://localhost:3000', 'X-Care-Client':'active-care-web'}


def jpeg(width=640, height=480):
    data = BytesIO()
    Image.new('RGB', (width, height)).save(data, format='JPEG')
    return data.getvalue()


def frame_headers(client, frame=1):
    session = client.post('/sessions', headers=HEADERS).json()['session_id']
    return {**HEADERS,'X-Session-Id':session,'X-Frame-Id':str(frame),'Content-Type':'image/jpeg'}


def test_health_and_real_contract_no_people(client):
    assert client.get('/health').json()['status'] == 'ready'
    headers = frame_headers(client)
    response = client.post('/frames', content=jpeg(), headers=headers)
    assert response.status_code == 200
    assert response.json()['persons'] == []
    assert response.json()['frame_id'] == 1
    assert response.json()['width'] == 640
    assert response.headers['cache-control'] == 'no-store'


def test_text_decision_is_lazy_local_and_reuses_model(client):
    status = client.get('/decision/status').json()
    assert status == {'available': True, 'loaded': False, 'run_name': 'test-double'}
    text = '地点：厨房。\n事件1：患者反复操作灶台旋钮。'
    first = client.post('/decision/predict', json={'text':text}, headers=HEADERS)
    second = client.post('/decision/predict', json={'text':text}, headers=HEADERS)
    assert first.status_code == second.status_code == 200
    assert first.json()['source'] == 'local_text_model'
    assert first.json()['guarded_output']['intervention_level'] == 'I2'
    assert FakeDecisionModel.created == 1
    assert client.get('/health').json()['decision_loaded'] is True


def test_text_decision_validates_length_and_client_marker(client):
    assert client.post('/decision/predict', json={'text':'太短'}, headers=HEADERS).status_code == 422
    assert client.post('/decision/predict', json={'text':'这是长度足够但没有客户端标记的文字输入。'}).status_code == 403


def test_reject_remote_origin_and_unmarked_post(client):
    assert client.post('/sessions', headers={**HEADERS,'Origin':'https://evil.example'}).status_code == 403
    assert client.post('/sessions').status_code == 403
    assert client.get('/health', headers={'Host':'evil.example'}).status_code == 400


def test_preflight_only_permits_local_origin(client):
    headers = {'Origin':'http://localhost:3000','Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'x-care-client,content-type,x-session-id,x-frame-id'}
    r = client.options('/frames', headers=headers)
    assert r.status_code == 200
    assert r.headers['access-control-allow-origin'] == 'http://localhost:3000'


@pytest.mark.parametrize('data', [b'not a jpeg', jpeg(1600, 1200), b'x'*1_500_001])
def test_reject_malformed_oversized_images(client, data):
    assert client.post('/frames', content=data, headers=frame_headers(client)).status_code in (400,413)


def test_out_of_order_and_closed_session(client):
    headers = frame_headers(client)
    assert client.post('/frames', content=jpeg(), headers=headers).status_code == 200
    assert client.post('/frames', content=jpeg(), headers=headers).status_code == 409
    client.delete('/sessions', headers=headers)
    headers['X-Frame-Id'] = '2'
    assert client.post('/frames', content=jpeg(), headers=headers).status_code == 410


def test_content_type_and_invalid_frame_id(client):
    headers = frame_headers(client)
    assert client.post('/frames', content=jpeg(), headers={**headers,'Content-Type':'text/plain'}).status_code == 415
    assert client.post('/frames', content=jpeg(), headers={**headers,'X-Frame-Id':'NaN'}).status_code == 400


def test_sessions_are_isolated_and_bounded(client):
    a, b = frame_headers(client), frame_headers(client)
    assert a['X-Session-Id'] != b['X-Session-Id']
    assert client.post('/frames', content=jpeg(), headers=a).status_code == 200
    assert client.post('/frames', content=jpeg(), headers=b).status_code == 200
    client.post('/sessions', headers=HEADERS)
    client.post('/sessions', headers=HEADERS)
    assert client.post('/sessions', headers=HEADERS).status_code == 429
