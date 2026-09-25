import asyncio
import copy
import logging
from types import SimpleNamespace

import pytest
from open_webui.models.users import UserSettings
from open_webui.utils import prefill
from open_webui.utils.prefill import PrefillStatus, apply_prefill_switch

OFFLOAD = {'available': True, 'mode': 'media', 'auto': True, 'break_even_tokens': 6000}
MODELS = {
    'mlx': {'id': 'mlx', 'prefill_offload': OFFLOAD},
    'mlx-no-auto': {'id': 'mlx-no-auto', 'prefill_offload': {**OFFLOAD, 'auto': False}},
    'plain': {'id': 'plain'},
}
PRESET = {'id': 'preset', 'info': {'base_model_id': 'mlx'}}
CHAT = {'chat_id': 'chat-1', 'message_id': 'message-1'}


def user_with(settings: dict | None):
    return SimpleNamespace(settings=None if settings is None else UserSettings.model_validate(settings))


def apply(model=MODELS['mlx'], settings=None, metadata=CHAT, form_data=None):
    return apply_prefill_switch(dict(form_data or {'model': model['id']}), model, user_with(settings), metadata, MODELS)


class TestApplyPrefillSwitch:
    @pytest.mark.parametrize(
        'settings, route',
        [
            ({'ui': {}, 'prefillOffload': {'mlx': True}}, 'auto'),
            ({'ui': {}, 'prefillOffload': {'mlx': False}}, 'local'),
            ({'ui': {}, 'prefillOffload': {'other': False}}, 'auto'),
            ({'ui': {}}, 'auto'),
            (None, 'auto'),
        ],
        ids=['on', 'off', 'missing-entry', 'no-switches', 'settings-none'],
    )
    def test_switch_sets_route_and_progress(self, settings, route):
        form_data = apply(settings=settings)
        assert form_data == {'model': 'mlx', 'prefill_route': route, 'prefill_progress': True}

    def test_temporary_chat_is_covered(self):
        form_data = apply(metadata={'chat_id': 'local:socket-1', 'message_id': 'message-1'})
        assert form_data['prefill_route'] == 'auto'

    def test_preset_uses_base_model_switch_and_capability(self):
        assert apply(PRESET, {'prefillOffload': {'mlx': False}})['prefill_route'] == 'local'
        assert apply(PRESET, {'prefillOffload': {'preset': False}})['prefill_route'] == 'auto'

    @pytest.mark.parametrize(
        'model',
        [MODELS['mlx-no-auto'], MODELS['plain'], {'id': 'preset', 'info': {'base_model_id': 'gone'}}],
        ids=['auto-false', 'no-prefill-offload', 'preset-base-not-served'],
    )
    def test_model_without_auto_offload_is_untouched(self, model):
        assert apply(model) == {'model': model['id']}

    @pytest.mark.parametrize(
        'metadata',
        [
            {'chat_id': '', 'message_id': None},
            {'chat_id': 'chat-1', 'message_id': None},
            {},
            {'chat_id': 'channel:channel-1', 'message_id': 'message-1'},
        ],
        ids=['api-call', 'no-message-id', 'no-metadata', 'channel'],
    )
    def test_request_outside_chat_ui_is_untouched(self, metadata):
        assert apply(metadata=metadata) == {'model': 'mlx'}

    def test_params_value_is_replaced_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger='open_webui.utils.prefill'):
            form_data = apply(form_data={'model': 'mlx', 'prefill_route': 'remote'})
        assert form_data['prefill_route'] == 'auto'
        [record] = caplog.records
        assert "prefill_route='remote'" in record.getMessage() and "with 'auto'" in record.getMessage()

    def test_same_params_value_is_kept_silently(self, caplog):
        with caplog.at_level(logging.WARNING, logger='open_webui.utils.prefill'):
            apply(form_data={'model': 'mlx', 'prefill_route': 'auto', 'prefill_progress': True})
        assert caplog.records == []


############################
# PrefillStatus
############################


def route(route, reason, notice=None):
    return {
        'object': 'chat.completion.chunk',
        'choices': [],
        'prefill_route': {'route': route, 'reason': reason, 'notice': notice},
    }


def progress(route, phase, percent, done=0, total=48):
    return {
        'object': 'chat.completion.chunk',
        'choices': [],
        'prefill_progress': {'percent': percent, 'route': route, 'phase': phase, 'done': done, 'total': total},
    }


def content(text='Hi'):
    return {'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'content': text}}]}


BUSY = 'Windows prefill worker is busy; prefilling on the Mac. Details: Prefill worker busy.'


def recorder():
    """An emitter and the list of events it received."""
    events = []

    async def emit(event):
        events.append(event)

    return emit, events


def run_stream(chunks):
    """Feed (time, chunk) pairs to one PrefillStatus: (the chunks it consumed, the events it sent to
    the saving emitter, the events it sent to the live one)."""
    clock = SimpleNamespace(now=0.0)
    save, saved = recorder()
    show, live = recorder()

    async def feed():
        status = PrefillStatus(save, CHAT, clock=lambda: clock.now, live_emitter=show)
        consumed = []
        for at, chunk in chunks:
            clock.now = at
            if await status.handle(chunk):
                consumed.append(chunk)
        return consumed

    return asyncio.run(feed()), saved, live


def lines(events):
    """Status descriptions, checking every status is a done 'prefill' one."""
    statuses = [event['data'] for event in events if event['type'] == 'status']
    assert all(status['action'] == 'prefill' and status['done'] is True for status in statuses)
    return [status['description'] for status in statuses]


def toasts(events):
    return [event['data'] for event in events if event['type'] == 'notification']


class TestPrefillStatus:
    def test_remote_shows_route_phases_steps_live_and_saves_final_line(self):
        consumed, saved, live = run_stream(
            [
                (0.0, route('remote', 'above_break_even')),
                (0.1, progress('remote', 'upload', 0.0)),
                (2.0, progress('remote', 'vision', 0.0)),
                (9.0, progress('remote', 'prefill', 0.0)),
                (10.0, progress('remote', 'prefill', 4.2, 2)),
                (60.0, progress('remote', 'prefill', 41.7, 20)),
                (61.0, progress('remote', 'prefill', 43.8, 21)),
                (170.0, progress('remote', 'prefill', 100.0, 48, 48)),
                (171.0, progress('remote', 'import', 100.0, 48, 48)),
                (181.2, content()),
                (181.3, content()),
            ]
        )
        assert len(consumed) == 9
        assert lines(live) == [
            'Prefilling on Windows',
            'Prefill on Windows: uploading',
            'Prefill on Windows: encoding media',
            'Prefill on Windows: 0%',
            'Prefill on Windows: 41%',
            'Prefill on Windows: 100%',
            'Prefill on Windows: loading result',
        ]
        assert saved == [
            {
                'type': 'status',
                'data': {'action': 'prefill', 'description': 'Prefill done on Windows · 181 s', 'done': True},
            }
        ]
        assert toasts(live) == []

    def test_local_notice_shows_live_toasts_once_and_saves_the_reason(self):
        _, saved, live = run_stream(
            [
                (0.5, route('local', 'worker_busy', BUSY)),
                (1.0, progress('local', 'prefill', 12.0)),
                (2.0, progress('local', 'prefill', 15.0)),
                (42.0, content()),
            ]
        )
        assert lines(live) == [BUSY, 'Prefill on Mac: 12%']
        assert lines(saved) == ['Prefill done on Mac · 42 s · Windows busy']
        # A toast is never saved; it goes out once, through the reply's emitter.
        assert toasts(saved) == [{'type': 'warning', 'content': BUSY}] and toasts(live) == []

    def test_unreachable_notice_gets_the_start_hint(self):
        notice = 'Windows prefill worker is not reachable; prefilling on the Mac.'
        _, saved, live = run_stream([(0.5, route('local', 'worker_unreachable', notice)), (5.0, content())])
        expected = f'{notice} {prefill.WORKER_UNREACHABLE_HINT}'
        assert lines(live) == [expected]
        assert toasts(saved) == [{'type': 'warning', 'content': expected}]
        assert lines(saved) == ['Prefill done on Mac · 5 s · Windows unreachable']

    def test_unknown_reason_with_notice_reads_generically(self):
        _, saved, live = run_stream([(0.0, route('local', 'new_reason', 'Something new.')), (5.0, content())])
        assert lines(live) == ['Something new.']
        assert lines(saved) == ['Prefill done on Mac · 5 s · Windows not used']

    def test_fast_local_prefix_cached_shows_nothing(self):
        consumed, saved, live = run_stream(
            [
                (0.2, route('local', 'prefix_cached')),
                (0.3, progress('local', 'prefill', 0.0)),
                (1.3, progress('local', 'prefill', 100.0)),
                (1.5, content()),
            ]
        )
        assert len(consumed) == 3
        assert saved == [] and live == []

    def test_slow_local_without_route_chunk_shows_after_quiet_time(self):
        _, saved, live = run_stream(
            [
                (1.0, progress('local', 'prefill', 5.0)),
                (2.0, progress('local', 'prefill', 15.0)),
                (3.5, progress('local', 'prefill', 25.0)),
                (4.5, progress('local', 'prefill', 27.0)),
                (6.0, progress('local', 'prefill', 31.0)),
                (40.0, content()),
            ]
        )
        assert lines(live) == ['Prefill on Mac: 25%', 'Prefill on Mac: 31%']
        assert lines(saved) == ['Prefill done on Mac · 40 s']

    @pytest.mark.parametrize(
        'chunk',
        [
            content(),
            {'object': 'chat.completion.chunk', 'choices': [], 'usage': {'prompt_tokens': 3}},
            {'type': 'response.output_text.delta', 'delta': 'Hi'},
            ['not', 'a', 'dict'],
        ],
        ids=['content', 'usage', 'responses-event', 'list'],
    )
    def test_other_chunks_pass_through_untouched(self, chunk):
        before = copy.deepcopy(chunk)
        consumed, saved, live = run_stream([(0.0, chunk)])
        assert consumed == [] and saved == [] and live == [] and chunk == before

    def test_error_chunk_passes_through_and_saves_nothing(self):
        error = {'error': {'message': 'Worker failed', 'type': 'prefill_error', 'code': 'prefill_failed'}}
        consumed, saved, live = run_stream([(0.0, route('remote', 'above_break_even')), (5.0, error)])
        assert consumed == [route('remote', 'above_break_even')]
        assert lines(live) == ['Prefilling on Windows'] and saved == []

    def test_responses_api_output_event_ends_the_prefill(self):
        responses_route = {'type': 'response.prefill_route', 'prefill_route': route('remote', 'x')['prefill_route']}
        consumed, saved, live = run_stream(
            [
                (0.0, {'type': 'response.created', 'response': {}}),
                (0.1, responses_route),
                (30.0, {'type': 'response.output_item.added', 'item': {'type': 'message'}}),
                (30.1, {'type': 'response.output_text.delta', 'delta': 'Hi'}),
            ]
        )
        assert consumed == [responses_route]
        assert lines(live) == ['Prefilling on Windows']
        assert lines(saved) == ['Prefill done on Windows · 30 s']

    def test_live_emitter_defaults_to_a_non_saving_socket_emitter_made_once(self, monkeypatch):
        show, live = recorder()
        calls = []

        async def get_event_emitter(request_info, update_db=True):
            calls.append((request_info, update_db))
            return show

        monkeypatch.setattr('open_webui.socket.main.get_event_emitter', get_event_emitter)
        save, saved = recorder()

        async def feed():
            status = PrefillStatus(save, CHAT)
            await status.handle(route('remote', 'above_break_even'))
            await status.handle(progress('remote', 'prefill', 50.0))
            await status.handle(content())

        asyncio.run(feed())
        assert calls == [(CHAT, False)]
        assert lines(live) == ['Prefilling on Windows', 'Prefill on Windows: 50%']
        assert lines(saved) == ['Prefill done on Windows · 0 s']

    @pytest.mark.parametrize(
        'signal, expected_lines, expected_toasts',
        [
            (route('local', 'worker_busy', BUSY)['prefill_route'], [BUSY], [{'type': 'warning', 'content': BUSY}]),
            (route('remote', 'above_break_even')['prefill_route'], ['Prefilling on Windows'], []),
            (None, [], []),
        ],
        ids=['local-notice', 'remote', 'no-route'],
    )
    def test_non_streaming_response_route_is_saved(self, signal, expected_lines, expected_toasts):
        save, saved = recorder()

        async def no_live(event):
            raise AssertionError(f'a non-streaming reply showed a line without saving it: {event}')

        response_data = {'choices': [{'message': {'content': 'Hi'}}]}
        if signal is not None:
            response_data['prefill_route'] = signal
        asyncio.run(PrefillStatus(save, CHAT, live_emitter=no_live).handle_response(response_data))
        assert response_data == {'choices': [{'message': {'content': 'Hi'}}]}
        assert lines(saved) == expected_lines and toasts(saved) == expected_toasts
