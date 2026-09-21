import base64
import json
import uuid
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

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


def make_bundle(user_id='user-1', num_frames=3, width=64, height=32, fps=2.0, duration=1.5) -> dict:
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


def frame_file(upload_dir: Path, meta: dict, index: int) -> Path:
    return upload_dir / 'video_frames' / meta['id'] / f'{index:04d}.jpg'


############################
# inject_video_parts — frames mode
############################


class TestInjectFrames:
    def test_path_transport_builds_full_content_and_pops_files(self, upload_dir):
        meta = make_bundle()
        image = {'type': 'image', 'url': 'data:image/png;base64,AAAA'}

        form = video.inject_video_parts(request_with([image, frames_item(meta)]), MODEL, USER)

        message = form['messages'][0]
        assert 'files' not in message
        assert message['content'][:2] == [
            {'type': 'text', 'text': 'what moves?'},
            {'type': 'image_url', 'image_url': {'url': image['url']}},
        ]
        part = message['content'][2]
        assert part['type'] == 'video_frames'
        assert part['video_frames']['fps'] == 2.0
        assert part['video_frames']['duration'] == 1.5
        assert part['video_frames']['frames'] == [str(frame_file(upload_dir, meta, i)) for i in range(3)]
        # n=3, T=2 -> G=2 -> G*T*h*w = 2*2*32*64
        assert form['video_pixels'] == {'max_pixels': 8192}

    def test_shared_path_replaces_upload_dir_prefix(self, upload_dir, monkeypatch):
        monkeypatch.setattr(video, 'VIDEO_SHARED_PATH', Path('/mnt/uploads'))
        meta = make_bundle(num_frames=1)

        form = video.inject_video_parts(request_with([frames_item(meta)]), MODEL, USER)

        frames = form['messages'][0]['content'][1]['video_frames']['frames']
        assert frames == [f'/mnt/uploads/video_frames/{meta["id"]}/0000.jpg']

    def test_data_uri_transport_inlines_frames(self, upload_dir, monkeypatch):
        monkeypatch.setattr(video, 'VIDEO_FRAMES_TRANSPORT', 'data_uri')
        meta = make_bundle(num_frames=2)

        form = video.inject_video_parts(request_with([frames_item(meta)]), MODEL, USER)

        frames = form['messages'][0]['content'][1]['video_frames']['frames']
        prefix = 'data:image/jpeg;base64,'
        assert len(frames) == 2
        assert all(frame.startswith(prefix) for frame in frames)
        assert base64.b64decode(frames[1][len(prefix) :]) == frame_file(upload_dir, meta, 1).read_bytes()

    def test_other_users_bundle_is_forbidden(self, upload_dir):
        meta = make_bundle(user_id=OTHER.id)

        with pytest.raises(HTTPException) as excinfo:
            video.inject_video_parts(request_with([frames_item(meta)]), MODEL, USER)
        assert excinfo.value.status_code == 403

    def test_admin_may_use_any_bundle(self, upload_dir):
        meta = make_bundle(user_id=OTHER.id)

        form = video.inject_video_parts(request_with([frames_item(meta)]), MODEL, ADMIN)

        assert form['messages'][0]['content'][1]['type'] == 'video_frames'

    def test_missing_bundle_is_not_found(self, upload_dir):
        meta = {**make_bundle(), 'id': str(uuid.uuid4())}

        with pytest.raises(HTTPException) as excinfo:
            video.inject_video_parts(request_with([frames_item(meta)]), MODEL, USER)
        assert excinfo.value.status_code == 404

    def test_video_pixels_is_max_over_bundles(self, upload_dir):
        small = make_bundle(num_frames=3, width=64, height=32)
        large = make_bundle(num_frames=7, width=640, height=352)
        # small: ceil(3/2)*2*64*32 = 8192 ; large: ceil(7/2)*2*640*352 = 1802240
        assert video.max_pixels_for([small, large], 2) == 1802240

        form = video.inject_video_parts(request_with([frames_item(small), frames_item(large)]), MODEL, USER)

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

        form = video.inject_video_parts(request_with([frames_item(meta)]), model, USER)

        assert 'video_pixels' not in form
        assert form['messages'][0]['content'][1]['type'] == 'video_frames'

    def test_existing_video_pixels_is_kept(self, upload_dir):
        meta = make_bundle()
        form_data = {**request_with([frames_item(meta)]), 'video_pixels': {'max_pixels': 1}}

        form = video.inject_video_parts(form_data, MODEL, USER)

        assert form['video_pixels'] == {'max_pixels': 1}

    def test_missing_temporal_patch_size_is_rejected(self, upload_dir):
        meta = make_bundle()
        model = {'video_input': {'pixels': {'per_request_pixels': True}}}

        with pytest.raises(HTTPException) as excinfo:
            video.inject_video_parts(request_with([frames_item(meta)]), model, USER)
        assert excinfo.value.status_code == 400


############################
# inject_video_parts — file mode
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

        form = video.inject_video_parts(request_with([file_item(file_id)]), MODEL, USER)

        message = form['messages'][0]
        assert 'files' not in message
        assert message['content'] == [
            {'type': 'text', 'text': 'what moves?'},
            {'type': 'video', 'video': str(upload_dir / f'{file_id}_clip.mp4')},
        ]
        assert 'video_pixels' not in form

    def test_shared_path_applies_to_files(self, upload_dir, monkeypatch):
        monkeypatch.setattr(video, 'VIDEO_SHARED_PATH', Path('/mnt/uploads'))
        file_id = str(uuid.uuid4())
        (upload_dir / f'{file_id}_clip.mp4').write_bytes(b'\x00mp4')

        form = video.inject_video_parts(request_with([file_item(file_id)]), MODEL, USER)

        assert form['messages'][0]['content'][1] == {'type': 'video', 'video': f'/mnt/uploads/{file_id}_clip.mp4'}

    def test_data_uri_transport_sends_video_url(self, upload_dir, monkeypatch):
        monkeypatch.setattr(video, 'VIDEO_FRAMES_TRANSPORT', 'data_uri')
        file_id = str(uuid.uuid4())
        (upload_dir / f'{file_id}_clip.mp4').write_bytes(b'\x00mp4')

        form = video.inject_video_parts(request_with([file_item(file_id)]), MODEL, USER)

        expected = 'data:video/mp4;base64,' + base64.b64encode(b'\x00mp4').decode()
        assert form['messages'][0]['content'][1] == {'type': 'video_url', 'video_url': {'url': expected}}

    def test_missing_file_is_not_found(self, upload_dir):
        with pytest.raises(HTTPException) as excinfo:
            video.inject_video_parts(request_with([file_item(str(uuid.uuid4()))]), MODEL, USER)
        assert excinfo.value.status_code == 404


############################
# inject_video_parts — untouched messages and text URLs
############################


class TestInjectUntouched:
    def test_message_without_video_is_left_for_upstream(self, upload_dir):
        image = {'type': 'image', 'url': 'data:image/png;base64,AAAA'}
        form_data = {
            'model': 'm',
            'messages': [
                {'role': 'user', 'content': 'hi', 'files': [image]},
                {'role': 'assistant', 'content': 'hello', 'files': [{'type': 'video', 'id': 'x'}]},
            ],
        }

        form = video.inject_video_parts(form_data, MODEL, USER)

        assert form['messages'][0] == {'role': 'user', 'content': 'hi', 'files': [image]}
        assert form['messages'][1] == {'role': 'assistant', 'content': 'hello', 'files': [{'type': 'video', 'id': 'x'}]}
        assert 'video_pixels' not in form

    def test_request_without_messages(self, upload_dir):
        assert video.inject_video_parts({'model': 'm'}, MODEL, USER) == {'model': 'm'}


class TestTextUrls:
    def test_file_url_token_becomes_video_part(self, upload_dir):
        form = video.inject_video_parts(request_with([], 'what moves? file:///tmp/clip.mp4'), MODEL, USER)

        assert form['messages'][0]['content'] == [
            {'type': 'text', 'text': 'what moves?'},
            {'type': 'video', 'video': '/tmp/clip.mp4'},
        ]

    def test_http_url_token_becomes_video_url_part(self, upload_dir):
        form = video.inject_video_parts(request_with([], 'https://example.com/a/clip.MP4 describe'), MODEL, USER)

        assert form['messages'][0]['content'] == [
            {'type': 'text', 'text': 'describe'},
            {'type': 'video_url', 'video_url': {'url': 'https://example.com/a/clip.MP4'}},
        ]

    def test_non_video_and_embedded_urls_are_left_alone(self, upload_dir):
        text = 'see https://example.com/page.html and (file:///x.mp4)'

        form = video.inject_video_parts(request_with([], text), MODEL, USER)

        assert form['messages'][0]['content'] == text

    def test_only_user_messages(self, upload_dir):
        form_data = {'messages': [{'role': 'assistant', 'content': 'file:///tmp/clip.mp4'}]}

        form = video.inject_video_parts(form_data, MODEL, USER)

        assert form['messages'][0]['content'] == 'file:///tmp/clip.mp4'

    def test_text_parts_of_list_content(self, upload_dir):
        form_data = {
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': 'file:///tmp/clip.mp4 what moves?'},
                        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}},
                    ],
                }
            ]
        }

        form = video.inject_video_parts(form_data, MODEL, USER)

        assert form['messages'][0]['content'] == [
            {'type': 'text', 'text': 'what moves?'},
            {'type': 'video', 'video': '/tmp/clip.mp4'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}},
        ]


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


def post_frames(client: TestClient, frames: list[bytes], meta) -> object:
    return client.post(
        '/api/v1/video/frames',
        files=[('frames', (f'{index}.jpg', data, 'image/jpeg')) for index, data in enumerate(frames)],
        data={'meta': meta if isinstance(meta, str) else json.dumps(meta)},
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
        assert isinstance(meta['created_at'], int)
        assert (directory / '0001.jpg').read_bytes() == jpeg_bytes()

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
