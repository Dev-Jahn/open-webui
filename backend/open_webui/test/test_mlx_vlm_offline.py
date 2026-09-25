import asyncio
import os
import socket
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from open_webui.constants import ERROR_MESSAGES
from open_webui.routers import prefill as prefill_router
from open_webui.utils import mlx_vlm_offline
from open_webui.utils.auth import get_verified_user
from open_webui.utils.mlx_vlm_offline import MLX_VLM_OFFLINE_HINT, is_reachable, offline_error, unknown_model_error

HINT = MLX_VLM_OFFLINE_HINT
USER = SimpleNamespace(id='user-1', role='user')


@pytest.fixture
def up():
    """URL of a listening port: the kernel accepts the connection even though nobody reads."""
    with socket.socket() as server:
        server.bind(('127.0.0.1', 0))
        server.listen()
        yield f'http://127.0.0.1:{server.getsockname()[1]}/v1'


@pytest.fixture
def down():
    """URL of a port nobody listens on: the connection is refused at once."""
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    return f'http://127.0.0.1:{port}/v1'


@pytest.mark.skipif('MLX_VLM_OFFLINE_HINT' in os.environ, reason='the hint is overridden in this environment')
def test_default_hint_is_the_owners_text():
    assert MLX_VLM_OFFLINE_HINT == (
        'mlx-vlm server is not running. Start it on the Mac: mlx-vlm-server start --windows '
        '(or without --windows for the Mac only).'
    )


def test_is_reachable(up, down):
    assert asyncio.run(is_reachable(up)) is True
    assert asyncio.run(is_reachable(down)) is False
    with pytest.raises(ValueError, match='has no host'):
        asyncio.run(is_reachable('/v1'))


# GET /api/v1/prefill/connection


def connection_client(monkeypatch, enabled: bool, urls: list[str], configs: dict) -> TestClient:
    async def get_openai_runtime_config():
        return enabled, urls, ['sk'] * len(urls), configs

    monkeypatch.setattr(mlx_vlm_offline, 'get_openai_runtime_config', get_openai_runtime_config)
    app = FastAPI()
    app.include_router(prefill_router.router, prefix='/api/v1/prefill')
    app.dependency_overrides[get_verified_user] = lambda: USER
    return TestClient(app)


@pytest.mark.parametrize(
    'enabled, urls, configs, reachable',
    [
        (True, ['up'], {}, True),
        (True, ['up', 'down'], {}, False),
        (True, ['down'], {'0': {'enable': True}}, False),
        (True, ['up', 'down'], {'1': {'enable': False}}, True),
        (True, ['down'], {'down': {'enable': False}}, True),
        (False, ['down'], {}, True),
        (True, [], {}, True),
    ],
    ids=['up', 'one-down', 'down-enabled', 'down-disabled', 'down-disabled-legacy-key', 'openai-off', 'none'],
)
def test_connection_state(monkeypatch, up, down, enabled, urls, configs, reachable):
    by_name = {'up': up, 'down': down}
    configs = {by_name.get(key, key): value for key, value in configs.items()}
    client = connection_client(monkeypatch, enabled, [by_name[url] for url in urls], configs)

    response = client.get('/api/v1/prefill/connection')

    assert response.status_code == 200
    assert response.json() == {'reachable': reachable, 'hint': None if reachable else HINT}


def test_connection_state_needs_a_signed_in_user(monkeypatch, down):
    client = connection_client(monkeypatch, True, [down], {})
    del client.app.dependency_overrides[get_verified_user]

    assert client.get('/api/v1/prefill/connection').status_code == 401


# The chat error hook (main.py process_chat)

NOT_FOUND = ERROR_MESSAGES.MODEL_NOT_FOUND()  # "Model '' was not found"
MODELS = {
    'mlx': {'id': 'mlx', 'urlIdx': 0},
    'preset': {'id': 'preset', 'preset': True, 'info': {'base_model_id': 'mlx'}},
    'orphan': {'id': 'orphan', 'preset': True, 'info': {'base_model_id': 'gone'}},
    'llama': {'id': 'llama', 'owned_by': 'ollama', 'urls': [0]},
}


def chat_error(monkeypatch, url: str, model: dict, error: str = NOT_FOUND) -> tuple[str, list[int]]:
    """(error text the chat shows, connection indexes looked up)."""
    looked_up = []

    async def get_openai_connection(idx):
        looked_up.append(idx)
        return [url][idx], 'sk', {}

    monkeypatch.setattr(mlx_vlm_offline, 'get_openai_connection', get_openai_connection)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(MODELS=MODELS)))
    return asyncio.run(offline_error(request, model, error)), looked_up


@pytest.mark.parametrize('model_id', ['mlx', 'preset'])
@pytest.mark.parametrize('error', [NOT_FOUND, ERROR_MESSAGES.SERVER_CONNECTION_ERROR])
def test_unreachable_connection_turns_the_error_into_the_hint(monkeypatch, down, model_id, error):
    assert chat_error(monkeypatch, down, MODELS[model_id], error) == (HINT, [0])


@pytest.mark.parametrize('model_id', ['mlx', 'preset'])
def test_reachable_connection_keeps_upstreams_error(monkeypatch, up, model_id):
    assert chat_error(monkeypatch, up, MODELS[model_id]) == (NOT_FOUND, [0])


@pytest.mark.parametrize('model_id', ['llama', 'orphan'])
def test_model_without_an_openai_connection_keeps_its_error(monkeypatch, down, model_id):
    assert chat_error(monkeypatch, down, MODELS[model_id]) == (NOT_FOUND, [])


def test_a_failing_check_keeps_the_error(monkeypatch, down):
    async def get_openai_connection(idx):
        raise IndexError('list index out of range')  # the connection was removed meanwhile

    monkeypatch.setattr(mlx_vlm_offline, 'get_openai_connection', get_openai_connection)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(MODELS=MODELS)))

    assert asyncio.run(offline_error(request, MODELS['mlx'], NOT_FOUND)) == NOT_FOUND


@pytest.mark.parametrize('url_idx', [0, '0'])  # the frontend's for...in makes it a string
def test_direct_connection_model_keeps_its_error(monkeypatch, down, url_idx):
    """A model from the user's own direct connection: its urlIdx does not index the admin's connections."""
    model = {'id': 'gpt-x', 'urlIdx': url_idx, 'direct': True}

    assert chat_error(monkeypatch, down, model) == (NOT_FOUND, [])


# The rejection hook (main.py chat_completion): upstream's "Model not found" before the chat starts

REJECTED = 'Model not found'


def rejected_error(monkeypatch, urls: list[str], error: str) -> tuple[str, int]:
    """(error text the rejected request carries, how often the connections were read)."""
    reads = []

    async def get_openai_runtime_config():
        reads.append(1)
        return True, urls, ['sk'] * len(urls), {}

    monkeypatch.setattr(mlx_vlm_offline, 'get_openai_runtime_config', get_openai_runtime_config)
    return asyncio.run(unknown_model_error(error)), len(reads)


def test_unknown_model_with_an_unreachable_connection_gets_the_hint(monkeypatch, up, down):
    assert rejected_error(monkeypatch, [up, down], REJECTED) == (HINT, 1)


def test_unknown_model_with_reachable_connections_keeps_upstreams_error(monkeypatch, up):
    assert rejected_error(monkeypatch, [up], REJECTED) == (REJECTED, 1)


def test_other_rejections_are_not_checked(monkeypatch, down):
    assert rejected_error(monkeypatch, [down], 'Error: boom') == ('Error: boom', 0)


def test_a_failing_rejection_check_keeps_the_error(monkeypatch):
    assert rejected_error(monkeypatch, ['/v1'], REJECTED) == (REJECTED, 1)  # a URL without a host
