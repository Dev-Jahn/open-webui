import copy
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import aiohttp
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from open_webui.routers import prefill as prefill_router
from open_webui.utils.auth import get_admin_user, get_current_user

ADMIN = SimpleNamespace(id='admin-1', role='admin')
USER = SimpleNamespace(id='user-1', role='user')

OFFLOAD = {'available': True, 'mode': 'media', 'auto': True, 'break_even_tokens': 6000}
MODELS = {
    'mlx': {'id': 'mlx', 'urlIdx': 0, 'prefill_offload': OFFLOAD},
    'plain': {'id': 'plain', 'urlIdx': 0},
    'preset': {'id': 'preset', 'preset': True, 'info': {'base_model_id': 'mlx'}},
    'orphan': {'id': 'orphan', 'preset': True, 'info': {'base_model_id': 'gone'}},
}
RESULT = {
    'route_min_tokens': {'before': 6000, 'after': 5700},
    'applied': True,
    'saved_to': '/tmp/worker.json',
    'points': [
        {'prompt_tokens': 4133, 'local_seconds': 9.8, 'offload_seconds': 14.2},
        {'prompt_tokens': 12400, 'local_seconds': 35.1, 'offload_seconds': 13.0},
    ],
    'elapsed_seconds': 80.3,
    'note': None,
}


class FakeServer(ThreadingHTTPServer):
    """What the handler reads: the replies to give and the requests seen."""

    requests: list[dict]
    delay: float
    reply: tuple[int, bytes, str]


class FakeMlxVlm(BaseHTTPRequestHandler):
    """Answers every POST with `server.reply` = (status, body, content type) after `server.delay`."""

    server: FakeServer

    def do_POST(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        self.server.requests.append({'path': self.path, 'headers': dict(self.headers), 'json': json.loads(body)})
        time.sleep(self.server.delay)
        status, reply, content_type = self.server.reply
        try:
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the client gave up (timeout test)

    def log_message(self, format, *args):
        pass


def error_body(message: str, type_: str, code: str) -> bytes:
    return json.dumps({'error': {'message': message, 'type': type_, 'code': code}}).encode()


@pytest.fixture
def mlx_vlm():
    server = FakeServer(('127.0.0.1', 0), FakeMlxVlm)
    server.requests, server.delay = [], 0.0
    server.reply = (200, json.dumps(RESULT).encode(), 'application/json')
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


def make_client(monkeypatch, url: str) -> TestClient:
    async def get_openai_connection(idx):
        assert idx == 0
        return url, 'sk-test', {}

    monkeypatch.setattr(prefill_router, 'get_openai_connection', get_openai_connection)
    app = FastAPI()
    app.include_router(prefill_router.router, prefix='/api/v1/prefill')
    app.state.MODELS = copy.deepcopy(MODELS)
    app.dependency_overrides[get_admin_user] = lambda: ADMIN
    return TestClient(app)


@pytest.fixture
def client(mlx_vlm, monkeypatch):
    return make_client(monkeypatch, f'http://127.0.0.1:{mlx_vlm.server_port}/v1/')


def calibrate(client: TestClient, **body):
    return client.post('/api/v1/prefill/calibrate', json={'model_id': 'mlx', **body})


@pytest.mark.parametrize('body, sent', [({}, True), ({'apply': True}, True), ({'apply': False}, False)])
def test_success_returns_the_answer_and_clears_the_model_cache(client, mlx_vlm, body, sent):
    response = calibrate(client, **body)

    assert response.status_code == 200
    assert response.json() == RESULT
    [request] = mlx_vlm.requests
    assert request['path'] == '/v1/prefill/calibrate'
    assert request['json'] == {'apply': sent}
    assert request['headers']['Authorization'] == 'Bearer sk-test'
    assert client.app.state.MODELS == {}


def test_note_is_passed_through(client, mlx_vlm):
    answer = {**RESULT, 'note': 'the lines do not cross between 4133 and 12400 tokens'}
    mlx_vlm.reply = (200, json.dumps(answer).encode(), 'application/json')

    assert calibrate(client).json() == answer


def test_preset_calibrates_its_base_model(client, mlx_vlm):
    response = calibrate(client, model_id='preset')

    assert response.status_code == 200
    assert len(mlx_vlm.requests) == 1


@pytest.mark.parametrize(
    'status, body, detail',
    [
        (409, error_body('Another request is running', 'server_busy', 'server_busy'), 'Another request is running'),
        (503, error_body('Worker down', 'x', 'prefill_worker_unreachable'), 'Worker down'),
        (503, error_body('Worker busy', 'x', 'prefill_worker_busy'), 'Worker busy'),
        (400, error_body('No worker', 'x', 'prefill_worker_not_configured'), 'No worker'),
        (
            404,
            b'{"detail":"Not Found"}',
            'mlx-vlm answered HTTP 404 without an error message: {"detail":"Not Found"}',
        ),
        (500, b'', 'mlx-vlm answered HTTP 500 without an error message: (empty body)'),
    ],
    ids=['busy', 'worker-unreachable', 'worker-busy', 'not-configured', 'no-message', 'empty-body'],
)
def test_errors_keep_status_and_message_and_the_cache(client, mlx_vlm, status, body, detail):
    mlx_vlm.reply = (status, body, 'application/json')

    response = calibrate(client)

    assert (response.status_code, response.json()) == (status, {'detail': detail})
    assert client.app.state.MODELS == MODELS


@pytest.mark.parametrize(
    'model_id, status, detail',
    [
        ('plain', 400, 'Model plain has no prefill_offload'),
        ('orphan', 404, 'Base model gone of orphan is not available'),
        ('missing', 404, 'Model missing is not available'),
    ],
)
def test_models_that_cannot_be_calibrated(client, mlx_vlm, model_id, status, detail):
    response = calibrate(client, model_id=model_id)

    assert response.status_code == status
    assert response.json()['detail'].startswith(detail)
    assert mlx_vlm.requests == []


def test_non_admin_is_refused(client, mlx_vlm):
    del client.app.dependency_overrides[get_admin_user]
    client.app.dependency_overrides[get_current_user] = lambda: USER

    assert calibrate(client).status_code == 401
    assert mlx_vlm.requests == []


def test_server_off_is_a_clear_error(monkeypatch):
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    url = f'http://127.0.0.1:{port}/v1'

    response = calibrate(make_client(monkeypatch, url))

    assert response.status_code == 502
    assert response.json()['detail'] == f'mlx-vlm is not reachable at {url}; is it running?'


def test_no_answer_in_time_is_a_timeout(client, mlx_vlm, monkeypatch):
    monkeypatch.setattr(prefill_router, 'TIMEOUT', aiohttp.ClientTimeout(total=0.2, sock_connect=10))
    mlx_vlm.delay = 1.0

    response = calibrate(client)

    assert response.status_code == 504
    assert response.json()['detail'].startswith('mlx-vlm did not answer within 0.2 s')
    assert client.app.state.MODELS == MODELS
