import base64
import copy
import json
import uuid
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from open_webui.routers import video as video_router
from open_webui.utils import video
from open_webui.utils.auth import get_verified_user
from PIL import Image

USER = SimpleNamespace(id='user-1', role='user')
OTHER = SimpleNamespace(id='user-2', role='user')
ADMIN = SimpleNamespace(id='admin-1', role='admin')

MODEL = {'video_input': {'supported': True, 'pixels': {'temporal_patch_size': 2, 'per_request_pixels': True}}}
MODEL_TIMESTAMPS = {'video_input': {**MODEL['video_input'], 'sampling': {'timestamps': True}}}


def jpeg_bytes(width: int = 64, height: int = 32) -> bytes:
    buffer = BytesIO()
    Image.new('RGB', (width, height), (200, 30, 30)).save(buffer, 'JPEG')
    return buffer.getvalue()


def png_bytes(width: int = 64, height: int = 32) -> bytes:
    buffer = BytesIO()
    Image.new('RGB', (width, height)).save(buffer, 'PNG')
    return buffer.getvalue()


@pytest.fixture
def upload_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(video, 'UPLOAD_DIR', tmp_path)
    monkeypatch.setattr(video, 'VIDEO_FRAMES_TRANSPORT', 'path')
    monkeypatch.setattr(video, 'VIDEO_SHARED_PATH', None)
    return tmp_path


def make_bundle(user_id='user-1', num_frames=3, width=64, height=32, fps=2.0, duration=1.5, **extra) -> dict:
    meta = {
        'id': str(uuid.uuid4()),
        'user_id': user_id,
        'created_at': 0,
        'fps': fps,
        'duration': duration,
        'width': width,
        'height': height,
        'num_frames': num_frames,
        'name': 'clip.mp4',
        'content_type': 'video/mp4',
        **extra,
    }
    video.write_bundle(meta['id'], [jpeg_bytes(width, height)] * num_frames, meta)
    return meta


def frames_item(meta: dict) -> dict:
    return {
        'type': 'video',
        'name': meta['name'],
        'content_type': 'video/mp4',
        'status': 'uploaded',
        'id': meta['id'],
        'video_frames': {k: meta[k] for k in ('id', 'num_frames', 'width', 'height', 'fps', 'duration')},
    }


def request_with(files: list[dict], text: str = 'what moves?') -> dict:
    return {'model': 'm', 'messages': [{'role': 'user', 'content': text, 'files': files}]}


IMAGE_A = {'type': 'image', 'url': 'data:image/png;base64,AAAA'}
IMAGE_B = {'type': 'file', 'content_type': 'image/jpeg', 'url': 'data:image/jpeg;base64,BBBB'}


def image_part(item: dict) -> dict:
    return {'type': 'image_url', 'image_url': {'url': item['url']}}


def frame_file(upload_dir: Path, meta: dict, index: int) -> Path:
    return upload_dir / 'video_frames' / meta['id'] / f'{index:04d}.jpg'


############################
# inject_media_parts — frames mode
############################


class TestInjectFrames:
    def test_path_transport_builds_full_content_and_pops_files(self, upload_dir):
        meta = make_bundle()
        image = {'type': 'image', 'url': 'data:image/png;base64,AAAA'}

        form = video.inject_media_parts(request_with([image, frames_item(meta)]), MODEL, USER, {})

        message = form['messages'][0]
        assert 'files' not in message
        image_part, part, text_part = message['content']
        assert image_part == {'type': 'image_url', 'image_url': {'url': image['url']}}
        assert text_part == {'type': 'text', 'text': 'what moves?'}
        assert part['type'] == 'video_frames'
        assert part['video_frames']['fps'] == 2.0
        assert part['video_frames']['duration'] == 1.5
        assert part['video_frames']['frames'] == [str(frame_file(upload_dir, meta, i)) for i in range(3)]
        # n=3, T=2 -> G=2 -> G*T*h*w = 2*2*32*64
        assert form['video_pixels'] == {'max_pixels': 8192}

    def test_shared_path_replaces_upload_dir_prefix(self, upload_dir, monkeypatch):
        monkeypatch.setattr(video, 'VIDEO_SHARED_PATH', Path('/mnt/uploads'))
        meta = make_bundle(num_frames=1)

        form = video.inject_media_parts(request_with([frames_item(meta)]), MODEL, USER, {})

        frames = form['messages'][0]['content'][0]['video_frames']['frames']
        assert frames == [f'/mnt/uploads/video_frames/{meta["id"]}/0000.jpg']

    def test_data_uri_transport_inlines_frames(self, upload_dir, monkeypatch):
        monkeypatch.setattr(video, 'VIDEO_FRAMES_TRANSPORT', 'data_uri')
        meta = make_bundle(num_frames=2)

        form = video.inject_media_parts(request_with([frames_item(meta)]), MODEL, USER, {})

        frames = form['messages'][0]['content'][0]['video_frames']['frames']
        prefix = 'data:image/jpeg;base64,'
        assert len(frames) == 2
        assert all(frame.startswith(prefix) for frame in frames)
        assert base64.b64decode(frames[1][len(prefix) :]) == frame_file(upload_dir, meta, 1).read_bytes()

    def test_other_users_bundle_is_forbidden(self, upload_dir):
        meta = make_bundle(user_id=OTHER.id)

        with pytest.raises(HTTPException) as excinfo:
            video.inject_media_parts(request_with([frames_item(meta)]), MODEL, USER, {})
        assert excinfo.value.status_code == 403

    def test_admin_may_use_any_bundle(self, upload_dir):
        meta = make_bundle(user_id=OTHER.id)

        form = video.inject_media_parts(request_with([frames_item(meta)]), MODEL, ADMIN, {})

        assert form['messages'][0]['content'][0]['type'] == 'video_frames'

    def test_missing_bundle_is_not_found(self, upload_dir):
        meta = {**make_bundle(), 'id': str(uuid.uuid4())}

        with pytest.raises(HTTPException) as excinfo:
            video.inject_media_parts(request_with([frames_item(meta)]), MODEL, USER, {})
        assert excinfo.value.status_code == 404

    def test_video_pixels_is_max_over_bundles(self, upload_dir):
        small = make_bundle(num_frames=3, width=64, height=32)
        large = make_bundle(num_frames=7, width=640, height=352)
        # small: ceil(3/2)*2*64*32 = 8192 ; large: ceil(7/2)*2*640*352 = 1802240
        assert video.max_pixels_for([small, large], 2) == 1802240

        form = video.inject_media_parts(request_with([frames_item(small), frames_item(large)]), MODEL, USER, {})

        assert form['video_pixels'] == {'max_pixels': 1802240}

    @pytest.mark.parametrize(
        'model',
        [
            {'video_input': {'pixels': {'temporal_patch_size': 2, 'per_request_pixels': False}}},
            {'video_input': {'pixels': None}},
            {},
        ],
    )
    def test_video_pixels_is_not_set_without_per_request_pixels(self, upload_dir, model):
        meta = make_bundle()

        form = video.inject_media_parts(request_with([frames_item(meta)]), model, USER, {})

        assert 'video_pixels' not in form
        assert form['messages'][0]['content'][0]['type'] == 'video_frames'

    def test_existing_video_pixels_is_kept(self, upload_dir):
        meta = make_bundle()
        form_data = {**request_with([frames_item(meta)]), 'video_pixels': {'max_pixels': 1}}

        form = video.inject_media_parts(form_data, MODEL, USER, {})

        assert form['video_pixels'] == {'max_pixels': 1}

    def test_missing_temporal_patch_size_is_rejected(self, upload_dir):
        meta = make_bundle()
        model = {'video_input': {'pixels': {'per_request_pixels': True}}}

        with pytest.raises(HTTPException) as excinfo:
            video.inject_media_parts(request_with([frames_item(meta)]), model, USER, {})
        assert excinfo.value.status_code == 400


############################
# inject_media_parts — frame timestamps and presets
############################


def frames_spec(form: dict) -> dict:
    return form['messages'][0]['content'][0]['video_frames']


class TestTimestamps:
    def test_stored_timestamps_are_sent_when_the_model_reads_them(self, upload_dir):
        meta = make_bundle(timestamps=[0.0, 0.7, 1.5])

        form = video.inject_media_parts(request_with([frames_item(meta)]), MODEL_TIMESTAMPS, USER, {})

        assert frames_spec(form)['timestamps'] == [0.0, 0.7, 1.5]
        assert (frames_spec(form)['fps'], frames_spec(form)['duration']) == (2.0, 1.5)

    def test_bundle_without_timestamps_sends_its_slot_centres(self, upload_dir):
        meta = make_bundle(num_frames=3, duration=1.5)

        form = video.inject_media_parts(request_with([frames_item(meta)]), MODEL_TIMESTAMPS, USER, {})

        # the old sampler took frame i at (i + 0.5) * D / n
        assert frames_spec(form)['timestamps'] == [0.25, 0.75, 1.25]

    @pytest.mark.parametrize(
        'model',
        [
            pytest.param(MODEL, id='no-sampling'),
            pytest.param({'video_input': {'sampling': {'timestamps': False}}}, id='timestamps-false'),
            pytest.param({}, id='no-video-input'),
        ],
    )
    def test_not_sent_unless_the_model_reads_them(self, upload_dir, model):
        meta = make_bundle(timestamps=[0.0, 0.7, 1.5])

        form = video.inject_media_parts(request_with([frames_item(meta)]), model, USER, {})

        assert 'timestamps' not in frames_spec(form)


PRESET = {'id': 'my-preset', 'preset': True, 'info': {'id': 'my-preset', 'base_model_id': 'base'}}


class TestPreset:
    @pytest.mark.parametrize(
        'model, models',
        [
            pytest.param(PRESET, {'base': MODEL_TIMESTAMPS}, id='preset-reads-its-base-model'),
            pytest.param({**MODEL_TIMESTAMPS, 'info': {'base_model_id': None}}, {}, id='edited-base-model'),
        ],
    )
    def test_video_input_is_resolved(self, upload_dir, model, models):
        meta = make_bundle()

        form = video.inject_media_parts(request_with([frames_item(meta)]), model, USER, models)

        assert form['video_pixels'] == {'max_pixels': 8192}
        assert frames_spec(form)['timestamps'] == [0.25, 0.75, 1.25]

    def test_preset_with_missing_base_model_is_rejected_for_frame_bundles(self, upload_dir):
        meta = make_bundle()

        with pytest.raises(HTTPException) as excinfo:
            video.inject_media_parts(request_with([frames_item(meta)]), PRESET, USER, {'other': MODEL})
        assert excinfo.value.status_code == 400
        assert "'base'" in excinfo.value.detail

    @pytest.mark.parametrize(
        'files, text, content_types',
        [
            pytest.param([], 'hi', None, id='text-only'),
            pytest.param([IMAGE_A], 'hi', ['image_url', 'text'], id='image'),
            pytest.param([], 'file:///tmp/clip.mp4 hi', ['video', 'text'], id='typed-video-url'),
        ],
    )
    def test_preset_with_missing_base_model_works_without_frame_bundles(self, upload_dir, files, text, content_types):
        """A preset whose base id is not a key of MODELS still reaches here through an Arena model
        (upstream refuses it earlier only when chosen directly): only frame bundles read video_input,
        so nothing else may fail on it."""
        form = video.inject_media_parts(request_with(files, text), PRESET, USER, {'other': MODEL})

        content = form['messages'][0]['content']
        if content_types is None:
            assert content == text
        else:
            assert [part['type'] for part in content] == content_types
        assert 'video_pixels' not in form


############################
# inject_media_parts — file mode
############################


def file_item(file_id: str) -> dict:
    return {
        'type': 'video',
        'name': 'clip.mp4',
        'content_type': 'video/mp4',
        'status': 'uploaded',
        'id': file_id,
        'url': file_id,
    }


class TestInjectFile:
    def test_path_transport_sends_local_path(self, upload_dir):
        file_id = str(uuid.uuid4())
        (upload_dir / f'{file_id}_clip.mp4').write_bytes(b'\x00mp4')

        form = video.inject_media_parts(request_with([file_item(file_id)]), MODEL, USER, {})

        message = form['messages'][0]
        assert 'files' not in message
        assert message['content'] == [
            {'type': 'video', 'video': str(upload_dir / f'{file_id}_clip.mp4')},
            {'type': 'text', 'text': 'what moves?'},
        ]
        assert 'video_pixels' not in form

    def test_shared_path_applies_to_files(self, upload_dir, monkeypatch):
        monkeypatch.setattr(video, 'VIDEO_SHARED_PATH', Path('/mnt/uploads'))
        file_id = str(uuid.uuid4())
        (upload_dir / f'{file_id}_clip.mp4').write_bytes(b'\x00mp4')

        form = video.inject_media_parts(request_with([file_item(file_id)]), MODEL, USER, {})

        assert form['messages'][0]['content'][0] == {'type': 'video', 'video': f'/mnt/uploads/{file_id}_clip.mp4'}

    def test_data_uri_transport_sends_video_url(self, upload_dir, monkeypatch):
        monkeypatch.setattr(video, 'VIDEO_FRAMES_TRANSPORT', 'data_uri')
        file_id = str(uuid.uuid4())
        (upload_dir / f'{file_id}_clip.mp4').write_bytes(b'\x00mp4')

        form = video.inject_media_parts(request_with([file_item(file_id)]), MODEL, USER, {})

        expected = 'data:video/mp4;base64,' + base64.b64encode(b'\x00mp4').decode()
        assert form['messages'][0]['content'][0] == {'type': 'video_url', 'video_url': {'url': expected}}

    def test_missing_file_is_not_found(self, upload_dir):
        with pytest.raises(HTTPException) as excinfo:
            video.inject_media_parts(request_with([file_item(str(uuid.uuid4()))]), MODEL, USER, {})
        assert excinfo.value.status_code == 404


############################
# inject_media_parts — untouched messages and text URLs
############################


class TestMixedModes:
    """One video_pixels budget per request: frame bundles and whole videos cannot share it."""

    @pytest.fixture
    def file_id(self, upload_dir):
        file_id = str(uuid.uuid4())
        (upload_dir / f'{file_id}_clip.mp4').write_bytes(b'\x00mp4')
        return file_id

    def test_frames_and_file_in_one_message(self, upload_dir, file_id):
        form_data = request_with([frames_item(make_bundle()), file_item(file_id)])

        with pytest.raises(HTTPException) as excinfo:
            video.inject_media_parts(form_data, MODEL, USER, {})
        assert excinfo.value.status_code == 400
        assert 'cannot be combined' in excinfo.value.detail

    def test_frames_in_history_and_file_in_the_new_message(self, upload_dir, file_id):
        form_data = {
            'messages': [
                {'role': 'user', 'content': 'first', 'files': [frames_item(make_bundle())]},
                {'role': 'assistant', 'content': 'a clip'},
                {'role': 'user', 'content': 'second', 'files': [file_item(file_id)]},
            ]
        }

        with pytest.raises(HTTPException) as excinfo:
            video.inject_media_parts(form_data, MODEL, USER, {})
        assert excinfo.value.status_code == 400

    def test_frames_and_a_video_url_in_the_text(self, upload_dir):
        form_data = request_with([frames_item(make_bundle())], 'compare https://example.com/clip.mp4')

        with pytest.raises(HTTPException) as excinfo:
            video.inject_media_parts(form_data, MODEL, USER, {})
        assert excinfo.value.status_code == 400

    def test_several_bundles_are_one_mode(self, upload_dir):
        form_data = request_with([frames_item(make_bundle()), frames_item(make_bundle())])

        form = video.inject_media_parts(form_data, MODEL, USER, {})

        assert [part['type'] for part in form['messages'][0]['content']] == ['video_frames', 'video_frames', 'text']


class TestInjectUntouched:
    def test_messages_without_media_are_left_for_upstream(self, upload_dir):
        doc = {'type': 'file', 'content_type': 'application/pdf', 'url': 'doc-id'}
        messages = [
            {'role': 'user', 'content': 'hi', 'files': [doc]},
            {'role': 'user', 'content': [{'type': 'text', 'text': 'a'}, {'type': 'text', 'text': 'b'}]},
            {'role': 'user', 'content': [{'type': 'text', 'text': 'c'}], 'files': [IMAGE_A]},
            {'role': 'user', 'content': None},
            {'role': 'assistant', 'content': 'hello', 'files': [IMAGE_A, {'type': 'video', 'id': 'x'}]},
        ]
        form_data = {'model': 'm', 'messages': copy.deepcopy(messages)}

        form = video.inject_media_parts(form_data, MODEL, USER, {})

        assert form == {'model': 'm', 'messages': messages}

    def test_request_without_messages(self, upload_dir):
        assert video.inject_media_parts({'model': 'm'}, MODEL, USER, {}) == {'model': 'm'}


class TestTextUrls:
    def test_file_url_token_becomes_video_part(self, upload_dir):
        form = video.inject_media_parts(request_with([], 'what moves? file:///tmp/clip.mp4'), MODEL, USER, {})

        assert form['messages'][0]['content'] == [
            {'type': 'video', 'video': '/tmp/clip.mp4'},
            {'type': 'text', 'text': 'what moves?'},
        ]

    def test_http_url_token_becomes_video_url_part(self, upload_dir):
        form = video.inject_media_parts(request_with([], 'https://example.com/a/clip.MP4 describe'), MODEL, USER, {})

        assert form['messages'][0]['content'] == [
            {'type': 'video_url', 'video_url': {'url': 'https://example.com/a/clip.MP4'}},
            {'type': 'text', 'text': 'describe'},
        ]

    def test_non_video_and_embedded_urls_are_left_alone(self, upload_dir):
        text = 'see https://example.com/page.html and (file:///x.mp4)'

        form = video.inject_media_parts(request_with([], text), MODEL, USER, {})

        assert form['messages'][0]['content'] == text

    def test_only_user_messages(self, upload_dir):
        form_data = {'messages': [{'role': 'assistant', 'content': 'file:///tmp/clip.mp4'}]}

        form = video.inject_media_parts(form_data, MODEL, USER, {})

        assert form['messages'][0]['content'] == 'file:///tmp/clip.mp4'


############################
# inject_media_parts — Qwen order: media first, text last
############################


class TestMediaOrder:
    def test_image_only_message_selects_images_as_upstream_does(self, upload_dir):
        doc = {'type': 'file', 'content_type': 'application/pdf', 'url': 'doc-id'}
        no_url = {'type': 'image', 'name': 'still uploading'}

        form = video.inject_media_parts(request_with([IMAGE_A, doc, no_url, IMAGE_B], 'compare\n'), MODEL, USER, {})

        message = form['messages'][0]
        assert message['content'] == [image_part(IMAGE_A), image_part(IMAGE_B), {'type': 'text', 'text': 'compare\n'}]
        assert 'files' not in message

    def test_images_and_videos_keep_attachment_order(self, upload_dir):
        first, second = make_bundle(), make_bundle()

        form = video.inject_media_parts(
            request_with([frames_item(first), IMAGE_A, frames_item(second), IMAGE_B]), MODEL, USER, {}
        )

        content = form['messages'][0]['content']
        assert [part['type'] for part in content] == ['video_frames', 'image_url', 'video_frames', 'image_url', 'text']
        assert first['id'] in content[0]['video_frames']['frames'][0]
        assert second['id'] in content[2]['video_frames']['frames'][0]
        assert (content[1], content[3]) == (image_part(IMAGE_A), image_part(IMAGE_B))

    def test_typed_video_urls_follow_the_attachments(self, upload_dir):
        form = video.inject_media_parts(request_with([IMAGE_A], 'describe file:///tmp/clip.mp4'), MODEL, USER, {})

        assert form['messages'][0]['content'] == [
            image_part(IMAGE_A),
            {'type': 'video', 'video': '/tmp/clip.mp4'},
            {'type': 'text', 'text': 'describe'},
        ]

    def test_list_content_keeps_its_media_first_and_its_text_last(self, upload_dir):
        """A list content already holds the images its sender meant: as upstream, attached images
        are not added to it again (no duplicate), attached videos are."""
        file_id = str(uuid.uuid4())
        (upload_dir / f'{file_id}_clip.mp4').write_bytes(b'\x00mp4')
        form_data = {
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': 'file:///tmp/clip.mp4 what moves?'},
                        image_part(IMAGE_B),
                        {'type': 'text', 'text': 'and why?'},
                    ],
                    'files': [IMAGE_B, file_item(file_id)],
                }
            ]
        }

        form = video.inject_media_parts(form_data, MODEL, USER, {})

        assert form['messages'][0]['content'] == [
            image_part(IMAGE_B),
            {'type': 'video', 'video': str(upload_dir / f'{file_id}_clip.mp4')},
            {'type': 'video', 'video': '/tmp/clip.mp4'},
            {'type': 'text', 'text': 'what moves?'},
            {'type': 'text', 'text': 'and why?'},
        ]

    def test_every_user_message_of_the_history(self, upload_dir):
        form_data = {
            'messages': [
                {'role': 'user', 'content': 'first', 'files': [IMAGE_A]},
                {'role': 'assistant', 'content': 'a picture'},
                {'role': 'user', 'content': 'second', 'files': [frames_item(make_bundle())]},
            ]
        }

        form = video.inject_media_parts(form_data, MODEL, USER, {})

        assert [[part['type'] for part in form['messages'][i]['content']] for i in (0, 2)] == [
            ['image_url', 'text'],
            ['video_frames', 'text'],
        ]


############################
# put_media_first — after upstream's add_file_context
############################

ATTACHED_FILES = {'type': 'text', 'text': '<attached_files>\n<file type="image" id="a"/>\n</attached_files>\n\n'}
FRAMES_PART = {'type': 'video_frames', 'video_frames': {'frames': ['/f/0000.jpg', '/f/0001.jpg'], 'fps': 2.0}}


class TestPutMediaFirst:
    def test_attached_files_text_moves_behind_the_media(self):
        question = {'type': 'text', 'text': 'what moves?'}
        messages = [{'role': 'user', 'content': [ATTACHED_FILES, image_part(IMAGE_A), FRAMES_PART, question]}]

        assert video.put_media_first(messages)[0]['content'] == [
            image_part(IMAGE_A),
            FRAMES_PART,
            ATTACHED_FILES,
            question,
        ]

    def test_text_parts_keep_their_order(self):
        texts = [{'type': 'text', 'text': name} for name in ('a', 'b', 'c')]
        content = [texts[0], image_part(IMAGE_A), texts[1], texts[2], image_part(IMAGE_B)]

        assert video.put_media_first([{'role': 'user', 'content': content}])[0]['content'] == [
            image_part(IMAGE_A),
            image_part(IMAGE_B),
            *texts,
        ]

    def test_string_content_and_other_roles_are_untouched(self):
        messages = [
            {'role': 'system', 'content': [ATTACHED_FILES, image_part(IMAGE_A)]},
            {'role': 'user', 'content': '<attached_files>\n</attached_files>\n\nhi'},
            {'role': 'assistant', 'content': [ATTACHED_FILES, image_part(IMAGE_A)]},
        ]
        expected = copy.deepcopy(messages)

        assert video.put_media_first(messages) == expected

    def test_idempotent_so_a_replayed_turn_is_byte_identical(self):
        messages = [{'role': 'user', 'content': [ATTACHED_FILES, image_part(IMAGE_A), FRAMES_PART, ATTACHED_FILES]}]

        once = json.dumps(video.put_media_first(copy.deepcopy(messages)))
        twice = json.dumps(video.put_media_first(video.put_media_first(copy.deepcopy(messages))))

        assert once == twice


def test_unknown_transport_is_rejected():
    with pytest.raises(ValueError, match='VIDEO_FRAMES_TRANSPORT'):
        video._parse_transport('ftp')


############################
# Router
############################

META = {'fps': 2.0, 'duration': 1.5, 'width': 64, 'height': 32, 'num_frames': 3, 'name': 'clip.mp4'}


@pytest.fixture
def client(upload_dir):
    app = FastAPI()
    app.include_router(video_router.router, prefix='/api/v1/video')
    app.dependency_overrides[get_verified_user] = lambda: USER
    return TestClient(app)


def act_as(client: TestClient, user) -> None:
    client.app.dependency_overrides[get_verified_user] = lambda: user


def post_frames(client: TestClient, frames: list[bytes], meta, timestamps=None) -> httpx.Response:
    data = {'meta': meta if isinstance(meta, str) else json.dumps(meta)}
    if timestamps is not None:
        data['timestamps'] = timestamps if isinstance(timestamps, str) else json.dumps(timestamps)
    return client.post(
        '/api/v1/video/frames',
        files=[('frames', (f'{index}.jpg', data, 'image/jpeg')) for index, data in enumerate(frames)],
        data=data,
    )


class TestRouter:
    def test_upload_stores_frames_and_meta(self, client, upload_dir):
        response = post_frames(client, [jpeg_bytes()] * 3, META)

        assert response.status_code == 200, response.text
        body = response.json()
        assert body == {'id': body['id'], 'num_frames': 3, 'width': 64, 'height': 32, 'fps': 2.0, 'duration': 1.5}
        directory = upload_dir / 'video_frames' / body['id']
        assert sorted(path.name for path in directory.iterdir()) == ['0000.jpg', '0001.jpg', '0002.jpg', 'meta.json']
        meta = json.loads((directory / 'meta.json').read_text())
        assert meta['id'] == body['id']
        assert meta['user_id'] == USER.id
        assert meta['name'] == 'clip.mp4'
        assert meta['content_type'] is None
        assert 'timestamps' not in meta
        assert isinstance(meta['created_at'], int)
        assert (directory / '0001.jpg').read_bytes() == jpeg_bytes()

    def test_upload_stores_timestamps(self, client, upload_dir):
        # integers are seconds too; the last frame may sit just past the rounded duration (1.5)
        response = post_frames(client, [jpeg_bytes()] * 3, META, [0, 0.75, 1.52])

        assert response.status_code == 200, response.text
        body = response.json()
        assert body['timestamps'] == [0.0, 0.75, 1.52]
        meta = json.loads((upload_dir / 'video_frames' / body['id'] / 'meta.json').read_text())
        assert meta['timestamps'] == [0.0, 0.75, 1.52]
        assert all(isinstance(t, float) for t in meta['timestamps'])

    @pytest.mark.parametrize(
        'timestamps',
        [
            pytest.param([0, 1.0], id='too-few'),
            pytest.param([-0.1, 0.5, 1.0], id='negative'),
            pytest.param([0, 1.0, 0.5], id='decreasing'),
            pytest.param([0, 0.75, 1.56], id='past-duration'),
            pytest.param('[0, NaN, 1]', id='nan'),
            pytest.param('[0, 1, Infinity]', id='infinity'),
            pytest.param('[0, true, 1]', id='boolean'),
            pytest.param('["0", 1, 1.2]', id='string'),
            pytest.param('{"0": 0}', id='not-a-list'),
            pytest.param('0, 1, 1.2', id='not-json'),
        ],
    )
    def test_bad_timestamps_are_400(self, client, upload_dir, timestamps):
        response = post_frames(client, [jpeg_bytes()] * 3, META, timestamps)

        assert response.status_code == 400, response.text
        assert response.json()['detail'].startswith('Invalid timestamps')
        assert not (upload_dir / 'video_frames').exists()

    @pytest.mark.parametrize(
        'frames, meta',
        [
            pytest.param([jpeg_bytes()] * 2, META, id='frame-count-mismatch'),
            pytest.param([], META, id='no-frames'),
            pytest.param([jpeg_bytes(), jpeg_bytes(32, 32), jpeg_bytes()], META, id='wrong-size'),
            pytest.param([jpeg_bytes(), b'not an image', jpeg_bytes()], META, id='undecodable'),
            pytest.param([jpeg_bytes(), png_bytes(), jpeg_bytes()], META, id='not-jpeg'),
            pytest.param([jpeg_bytes()] * 3, {**META, 'fps': 0}, id='fps-zero'),
            pytest.param([jpeg_bytes()] * 3, {**META, 'duration': -1}, id='negative-duration'),
            pytest.param([jpeg_bytes()], {**META, 'num_frames': 0}, id='num-frames-zero'),
            pytest.param([jpeg_bytes()] * 3, 'not json', id='meta-not-json'),
        ],
    )
    def test_bad_input_is_400(self, client, upload_dir, frames, meta):
        response = post_frames(client, frames, meta)

        assert response.status_code == 400, response.text
        assert not (upload_dir / 'video_frames').exists()

    def test_get_frame(self, client):
        bundle_id = post_frames(client, [jpeg_bytes()] * 3, META).json()['id']

        response = client.get(f'/api/v1/video/frames/{bundle_id}/2')

        assert response.status_code == 200
        assert response.headers['content-type'] == 'image/jpeg'
        assert response.content == jpeg_bytes()

    @pytest.mark.parametrize('index', [-1, 3])
    def test_get_frame_out_of_range_is_404(self, client, index):
        bundle_id = post_frames(client, [jpeg_bytes()] * 3, META).json()['id']

        assert client.get(f'/api/v1/video/frames/{bundle_id}/{index}').status_code == 404

    def test_get_frame_of_unknown_or_malformed_bundle_is_404(self, client):
        assert client.get(f'/api/v1/video/frames/{uuid.uuid4()}/0').status_code == 404
        assert client.get('/api/v1/video/frames/not-a-uuid/0').status_code == 404

    def test_get_frame_hides_other_users_bundles(self, client):
        bundle_id = post_frames(client, [jpeg_bytes()] * 3, META).json()['id']

        act_as(client, OTHER)
        assert client.get(f'/api/v1/video/frames/{bundle_id}/0').status_code == 404
        act_as(client, ADMIN)
        assert client.get(f'/api/v1/video/frames/{bundle_id}/0').status_code == 200

    def test_delete_removes_directory(self, client, upload_dir):
        bundle_id = post_frames(client, [jpeg_bytes()] * 3, META).json()['id']

        response = client.delete(f'/api/v1/video/frames/{bundle_id}')

        assert response.status_code == 200
        assert not (upload_dir / 'video_frames' / bundle_id).exists()
        assert client.delete(f'/api/v1/video/frames/{bundle_id}').status_code == 404

    def test_delete_hides_other_users_bundles(self, client, upload_dir):
        bundle_id = post_frames(client, [jpeg_bytes()] * 3, META).json()['id']

        act_as(client, OTHER)
        assert client.delete(f'/api/v1/video/frames/{bundle_id}').status_code == 404
        assert (upload_dir / 'video_frames' / bundle_id).is_dir()
