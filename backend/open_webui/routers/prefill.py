"""Measure mlx-vlm's local-vs-offload prefill break-even and apply it (admin only).

mlx-vlm's ``POST /v1/prefill/calibrate`` prefills a short and a long prompt locally and on the
prefill offload worker, finds where the two timings cross and, with ``apply``, makes that its
routing threshold, which ``/v1/models`` then reports as ``prefill_offload.break_even_tokens``. It
takes a minute or two; chats sent meanwhile wait behind it on the server. On an error nothing
changes.

``GET /connection`` (any verified user) says whether the OpenAI connections are reachable and, when
one is not, how to start mlx-vlm; ``MlxVlmOfflineHint`` under the chat input asks when the model
list is empty.
"""

import logging

import aiohttp
from fastapi import APIRouter, Depends, HTTPException, Request
from open_webui.env import AIOHTTP_CLIENT_SESSION_SSL
from open_webui.routers.openai import clear_openai_model_cache, get_headers_and_cookies, get_openai_connection
from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.json_codec import JSONCodec
from open_webui.utils.mlx_vlm_offline import MLX_VLM_OFFLINE_HINT, unreachable_connections
from open_webui.utils.models import get_all_models
from open_webui.utils.prefill import offload_entry
from pydantic import BaseModel

log = logging.getLogger(__name__)

router = APIRouter()

# The measurement takes a minute or two and may queue behind a running chat; a host that never
# accepts the connection fails after sock_connect instead of the whole budget.
TIMEOUT = aiohttp.ClientTimeout(total=600, sock_connect=10)


class CalibrateForm(BaseModel):
    model_id: str
    apply: bool = True


def _error_message(status: int, body: str) -> str:
    """mlx-vlm's ``error.message``; a body without one is returned raw and labelled as such."""
    try:
        return JSONCodec.loads(body)['error']['message']
    except (ValueError, TypeError, KeyError):
        return f'mlx-vlm answered HTTP {status} without an error message: {body.strip() or "(empty body)"}'


def _served_entry(models, model_id: str) -> tuple[str, dict]:
    """(served model id, its models entry) for `model_id`, a preset read as its base model, as
    utils/prefill.py does; an HTTPException when that model cannot be calibrated."""
    model = models.get(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f'Model {model_id} is not available')
    served_id, entry = offload_entry(model, models)
    if entry is None:
        raise HTTPException(status_code=404, detail=f'Base model {served_id} of {model_id} is not available')
    if 'prefill_offload' not in entry:
        raise HTTPException(
            status_code=400,
            detail=f'Model {served_id} has no prefill_offload (not an mlx-vlm with a prefill offload worker)',
        )
    if 'urlIdx' not in entry:
        raise HTTPException(status_code=400, detail=f'Model {served_id} is not served through an OpenAI connection')
    return served_id, entry


@router.post('/calibrate')
async def calibrate(request: Request, form_data: CalibrateForm, user=Depends(get_admin_user)) -> dict:
    """Run the calibration on the mlx-vlm serving `model_id` (a preset: its base model) and return
    mlx-vlm's answer unchanged. mlx-vlm's errors keep their status code and message."""
    if not request.app.state.MODELS:
        await get_all_models(request, user=user)
    model_id, entry = _served_entry(request.app.state.MODELS, form_data.model_id)

    url, key, api_config = await get_openai_connection(entry['urlIdx'])
    headers, cookies = await get_headers_and_cookies(request, url, key, api_config, user=user)
    try:
        async with (
            aiohttp.ClientSession(timeout=TIMEOUT, trust_env=True) as session,
            session.post(
                f'{url.rstrip("/")}/prefill/calibrate',
                json={'apply': form_data.apply},
                headers=headers,
                cookies=cookies,
                ssl=AIOHTTP_CLIENT_SESSION_SSL,
            ) as response,
        ):
            if response.status != 200:
                raise HTTPException(
                    status_code=response.status, detail=_error_message(response.status, await response.text())
                )
            result = await response.json(loads=JSONCodec.loads)
    except (aiohttp.ClientConnectorError, aiohttp.ConnectionTimeoutError) as e:
        log.warning('Prefill calibration: cannot reach %s: %s', url, e)
        raise HTTPException(status_code=502, detail=f'mlx-vlm is not reachable at {url}; is it running?')
    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f'mlx-vlm did not answer within {TIMEOUT.total:g} s; it may still finish and apply the measurement',
        )
    except aiohttp.ClientError as e:
        raise HTTPException(status_code=502, detail=f'The calibration request to mlx-vlm at {url} failed: {e}')

    log.info('Prefill calibration of %s (apply=%s): %s', model_id, form_data.apply, result.get('route_min_tokens'))
    # The models list is cached; drop it so the next fetch shows the new break_even_tokens.
    await clear_openai_model_cache(request)
    return result


@router.get('/connection')
async def connection(user=Depends(get_verified_user)) -> dict:
    """``reachable``: every enabled OpenAI connection accepts a TCP connection within a second or
    two; ``hint``: how to start mlx-vlm when one does not, else None."""
    unreachable = await unreachable_connections()
    if unreachable:
        log.info('OpenAI connections not reachable: %s', ', '.join(unreachable))
    return {'reachable': not unreachable, 'hint': MLX_VLM_OFFLINE_HINT if unreachable else None}
