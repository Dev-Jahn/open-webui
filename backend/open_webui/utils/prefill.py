"""Prefill offload for mlx-vlm: the per-model switch and the prefill status lines.

mlx-vlm can hand a long prompt's prefill to another machine, the prefill offload worker, and
decode locally. Its `/v1/models` entries say so in ``prefill_offload`` (``auto`` true: a request
with ``prefill_route: "auto"`` may be offloaded). Each user has one switch per model, the
top-level user setting ``prefillOffload`` = {model id: bool}; a missing entry means on. A preset
(``info.base_model_id``) uses its base model's switch and capability.

``apply_prefill_switch`` turns the switch into the request fields ``prefill_route`` ("auto" on,
"local" off) and ``prefill_progress``. ``PrefillStatus`` turns the ``prefill_route`` and
``prefill_progress`` chunks of one stream, or a non-streaming response's ``prefill_route``, into
chat status lines. The shapes are defined in mlx-vlm's ``mlx_vlm/server/prefill_signals.py``.
"""

import logging
import os
import time

from open_webui.utils.chat_id import CHANNEL_CHAT_ID_PREFIX

log = logging.getLogger(__name__)

# A local prefill without a notice stays silent this long, so short prompts never flicker.
QUIET_SECONDS = 3.0

# Line prefixes by mlx-vlm route: 'remote' is the offload worker, 'local' the server itself. A route
# mlx-vlm may add later shows under its own name.
PROGRESS_LABELS = {'remote': 'Prefill (offload worker)', 'local': 'Prefill (local)'}
DONE_LABELS = {'remote': 'Prefill offloaded', 'local': 'Prefill done locally'}
PHASES = {'vision': 'encoding media', 'upload': 'uploading', 'import': 'loading result'}
# The final line's short reason after a notice (the notice itself is the status line above it and
# the toast); any other reason reads 'offload not used'.
SHORT_REASONS = {'worker_busy': 'offload worker busy', 'worker_unreachable': 'offload worker unreachable'}
# Appended to the worker_unreachable notice (mlx-vlm's text stays deployment-neutral); empty turns it off.
WORKER_UNREACHABLE_HINT = os.environ.get(
    'PREFILL_WORKER_UNREACHABLE_HINT',
    'To use prefill offload: run prefill-worker start on the offload worker, or mlx-vlm-server start --offload.',
)


def offload_entry(model: dict, models: dict) -> tuple[str, dict | None]:
    """(switch key, models entry whose prefill_offload applies). A preset reads its base model's
    entry in `models` (request.app.state.MODELS): None when the base model is not served."""
    base_model_id = (model.get('info') or {}).get('base_model_id')
    if base_model_id:
        return base_model_id, models.get(base_model_id)
    return model['id'], model


def apply_prefill_switch(form_data: dict, model: dict, user, metadata: dict, models: dict) -> dict:
    """Set ``prefill_route`` from the user's switch and ask for ``prefill_progress``.

    Acts only on chat UI requests (metadata has chat_id and message_id; not a channel, whose
    emitter drops status lines and toasts) to a model whose (base) ``prefill_offload.auto`` is
    true; every other request is returned untouched. The switch owns both fields: a different
    value from the model or chat params (merged into form_data before this) is replaced, with a
    warning.
    """
    chat_id = metadata.get('chat_id') or ''
    if not (chat_id and metadata.get('message_id')) or chat_id.startswith(CHANNEL_CHAT_ID_PREFIX):
        return form_data
    model_id, entry = offload_entry(model, models)
    if entry is None or (entry.get('prefill_offload') or {}).get('auto') is not True:
        return form_data
    switches = getattr(user.settings, 'prefillOffload', None) or {}
    fields = {'prefill_route': 'local' if switches.get(model_id) is False else 'auto', 'prefill_progress': True}
    for key, value in fields.items():
        if key in form_data and form_data[key] != value:
            log.warning(
                'Replacing %s=%r from the params with %r: the prefill switch owns it', key, form_data[key], value
            )
    form_data.update(fields)
    return form_data


class PrefillStatus:
    """One reply's prefill signals as status lines (action 'prefill', all done so none keeps shimmering).

    A notice becomes a status line and a warning toast; a remote route is announced at once.
    Progress shows on each phase change and 10-percent step, but for a local route without a
    notice only after QUIET_SECONDS. When the answer starts after anything was shown, a last line
    says whether the prefill was offloaded and how long it took, timed from this object's creation.

    Only that last line is saved, through `event_emitter`: it is the reply's record and names a
    notice's short reason. The lines before it are shown live through `live_emitter` (by default
    ``get_event_emitter(metadata, update_db=False)``, made on first use) because saving a status
    rewrites the whole chat row, so a save during the prefill can erase the title that the
    concurrent title task writes. A non-streaming reply's one line (notice or route) is saved.
    """

    def __init__(self, event_emitter, metadata: dict, clock=time.monotonic, live_emitter=None):
        self.event_emitter = event_emitter
        self.metadata = metadata
        self.live_emitter = live_emitter
        self.clock = clock
        self.started = clock()
        self.route = None  # mlx-vlm's route ('remote' or 'local') once a chunk said so
        self.reason = None  # short reason once a notice was shown
        self.shown = None  # (phase, 10-percent step) of the last progress line
        self.emitted = False
        self.finished = False

    async def handle(self, data) -> bool:
        """Whether `data` is a prefill chunk, consumed here; any other chunk passes through untouched."""
        if not isinstance(data, dict):
            return False
        # The answer starts: a chat-completions chunk with choices, or a Responses API output event.
        if data.get('choices') or str(data.get('type', '')).startswith('response.output'):
            if self.emitted and not self.finished:
                await self._finish()
            return False
        if 'prefill_route' in data:
            await self._route(data['prefill_route'], save=False)
            return True
        if 'prefill_progress' in data:
            await self._progress(data['prefill_progress'])
            return True
        return False

    async def handle_response(self, response_data: dict) -> None:
        """Show and save a non-streaming response's top-level ``prefill_route`` (removed from it)."""
        if 'prefill_route' in response_data:
            await self._route(response_data.pop('prefill_route'), save=True)

    async def _route(self, route: dict, save: bool) -> None:
        self.route = route['route']
        notice = route.get('notice')
        if notice and route.get('reason') == 'worker_unreachable' and WORKER_UNREACHABLE_HINT:
            notice = f'{notice} {WORKER_UNREACHABLE_HINT}'
        if notice:
            self.reason = SHORT_REASONS.get(route.get('reason'), 'offload not used')
            await self._status(notice, save)
            await self.event_emitter({'type': 'notification', 'data': {'type': 'warning', 'content': notice}})
        if route['route'] == 'remote':
            await self._status('Prefill offloading', save)

    async def _progress(self, progress: dict) -> None:
        self.route = progress['route']
        phase, percent = progress['phase'], progress['percent']
        quiet = progress['route'] != 'remote' and self.reason is None and self.clock() - self.started < QUIET_SECONDS
        step = (phase, int(percent // 10))
        if quiet or step == self.shown:
            return
        self.shown = step
        label = f'{int(percent)}%' if phase == 'prefill' else PHASES.get(phase, phase)
        prefix = PROGRESS_LABELS.get(self.route, f'Prefill ({self.route})')
        await self._status(f'{prefix}: {label}', save=False)

    async def _finish(self) -> None:
        self.finished = True
        done = DONE_LABELS.get(self.route, f'Prefill done ({self.route})')
        line = f'{done} · {self.clock() - self.started:.0f} s'
        await self._status(f'{line} · {self.reason}' if self.reason else line, save=True)

    async def _status(self, description: str, save: bool) -> None:
        self.emitted = True
        emit = self.event_emitter if save else await self._live()
        await emit({'type': 'status', 'data': {'action': 'prefill', 'description': description, 'done': True}})

    async def _live(self):
        """The emitter that shows a line without saving it."""
        if self.live_emitter is None:
            # Imported here so this module stays importable without the socket server.
            from open_webui.socket.main import get_event_emitter

            self.live_emitter = await get_event_emitter(self.metadata, update_db=False)
        return self.live_emitter
