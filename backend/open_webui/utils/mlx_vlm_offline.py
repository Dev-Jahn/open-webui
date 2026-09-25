"""Tell the user how to start mlx-vlm when its OpenAI connection cannot be reached.

The owner starts mlx-vlm only when needed. While it is off, upstream answers a chat with "Model ''
was not found" (routers/openai.py found no models on the connection) or "Server Connection Error"
(its model list was still cached), or, when the backend lists no models at all (after a restart),
rejects it with "Model not found"; a fresh model list is empty ("Select a model"). The hint text
lives only here (env ``MLX_VLM_OFFLINE_HINT``); it reaches the chat through ``offline_error`` (the
hook in main.py's ``process_chat``) and ``unknown_model_error`` (the hook in ``chat_completion``'s
rejection), and the empty model list through ``GET /api/v1/prefill/connection``
(routers/prefill.py), which ``MlxVlmOfflineHint.svelte`` under the chat input asks.

Reachable means a TCP connection to the connection URL's host and port opens within
PROBE_TIMEOUT_SECONDS. The HTTP answer is not awaited, so a server that is up but busy still counts
as reachable; a stopped local server refuses at once. Proxy settings are not applied to the probe.
"""

import asyncio
import logging
import os
from urllib.parse import urlparse

from open_webui.routers.openai import get_openai_connection, get_openai_runtime_config

log = logging.getLogger(__name__)

MLX_VLM_OFFLINE_HINT = os.environ.get(
    'MLX_VLM_OFFLINE_HINT',
    'mlx-vlm server is not running. Start it on the Mac: mlx-vlm-server start --windows '
    '(or without --windows for the Mac only).',
)
PROBE_TIMEOUT_SECONDS = 1.5


async def is_reachable(url: str) -> bool:
    """Whether a TCP connection to `url`'s host and port (default 80, https 443) opens in time."""
    parsed = urlparse(url)
    if not parsed.hostname:
        raise ValueError(f'Connection URL {url!r} has no host')
    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(parsed.hostname, port), PROBE_TIMEOUT_SECONDS)
    except (OSError, TimeoutError):
        return False
    writer.close()
    return True


async def unreachable_connections() -> list[str]:
    """URLs of the enabled OpenAI connections that are not reachable (none while the OpenAI API is off)."""
    enabled, urls, _, configs = await get_openai_runtime_config()
    if not enabled:
        return []
    urls = [url for idx, url in enumerate(urls) if configs.get(str(idx), configs.get(url, {})).get('enable', True)]
    reachable = await asyncio.gather(*(is_reachable(url) for url in urls))
    return [url for url, ok in zip(urls, reachable) if not ok]


async def offline_error(request, model: dict, error: str) -> str:
    """A failed chat's error text: the offline hint when the OpenAI connection serving `model` (a
    preset: its base model) is unreachable, else `error` unchanged, as for a model not served
    through an OpenAI connection or through one of the user's own direct connections. Runs only
    after a chat failed, so normal requests never wait for the probe. A failure of the check itself
    is logged and keeps `error`: this hook must never stop upstream from reporting the chat's error."""
    if model.get('direct'):
        return error  # its urlIdx indexes the user's own connections, not the admin's
    try:
        base_model_id = (model.get('info') or {}).get('base_model_id')
        entry = request.app.state.MODELS.get(base_model_id) if base_model_id else model
        if entry is None or 'urlIdx' not in entry:
            return error
        url, _, _ = await get_openai_connection(entry['urlIdx'])
        if await is_reachable(url):
            return error
    except Exception:
        log.exception('Checking whether the connection of %s is reachable failed; keeping its error', model.get('id'))
        return error
    log.warning(
        '%s failed while its connection %s is unreachable (%s); replying with the offline hint',
        model.get('id'),
        url,
        error,
    )
    return MLX_VLM_OFFLINE_HINT


async def unknown_model_error(error: str) -> str:
    """The error text of a chat request rejected before it started: the offline hint when it is
    upstream's "Model not found" and an enabled OpenAI connection is unreachable, else `error`
    unchanged. Upstream says so when the backend does not list the model (its list stays empty
    after a restart while mlx-vlm is off) or its base model, and when the user lacks access; the
    connection behind the model is unknown then, so any unreachable one counts. Like
    ``offline_error``, a failure of the check is logged and keeps `error`."""
    if error != 'Model not found':
        return error
    try:
        unreachable = await unreachable_connections()
    except Exception:
        log.exception('Checking whether the OpenAI connections are reachable failed; keeping %r', error)
        return error
    if not unreachable:
        return error
    log.warning('Chat rejected with %r while %s is unreachable; replying with the offline hint', error, unreachable)
    return MLX_VLM_OFFLINE_HINT
