"""Video input for LLM servers that accept video content parts (mlx-vlm / vllm-mlx).

Frame bundles (uniformly sampled JPEG frames extracted in the browser) live under
``UPLOAD_DIR / 'video_frames' / <id> / {0000.jpg, 0001.jpg, ..., meta.json}`` and are
created by ``routers/video.py``. ``inject_video_parts`` expands the video items attached to
user messages into the parts the LLM server understands right before the request is sent.

Environment:
    VIDEO_FRAMES_TRANSPORT  'path' (default) or 'data_uri' — how frames / files reach the server.
    VIDEO_SHARED_PATH       UPLOAD_DIR as seen by the LLM server (e.g. a Docker bind mount);
                            when set it replaces the UPLOAD_DIR prefix of every path sent.
"""

import base64
import json
import logging
import math
import mimetypes
import os
import re
from pathlib import Path

from fastapi import HTTPException, status
from open_webui.config import UPLOAD_DIR

log = logging.getLogger(__name__)

TRANSPORTS = ('path', 'data_uri')
BUNDLE_DIRNAME = 'video_frames'
META_FILENAME = 'meta.json'
FRAME_CONTENT_TYPE = 'image/jpeg'

UUID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')
# A bare `file:///abs/clip.mp4` or `http(s)://host/clip.mp4` token in the user's text.
VIDEO_URL_RE = re.compile(r'(?<!\S)(?:https?://|file://)\S+\.(?:mp4|webm|mov|mkv|avi)(?!\S)', re.IGNORECASE)


def _parse_transport(value: str) -> str:
    if value not in TRANSPORTS:
        raise ValueError(f'VIDEO_FRAMES_TRANSPORT must be one of {TRANSPORTS}, got {value!r}')
    return value


def _parse_shared_path(value: str) -> Path | None:
    return Path(value) if value else None


VIDEO_FRAMES_TRANSPORT = _parse_transport(os.environ.get('VIDEO_FRAMES_TRANSPORT', 'path'))
VIDEO_SHARED_PATH = _parse_shared_path(os.environ.get('VIDEO_SHARED_PATH', ''))


############################
# Bundle storage
############################


def bundle_dir(bundle_id: str) -> Path:
    if not UUID_RE.match(bundle_id):
        raise ValueError(f'Invalid video frames bundle id: {bundle_id!r}')
    return Path(UPLOAD_DIR) / BUNDLE_DIRNAME / bundle_id


def frame_path(bundle_id: str, index: int) -> Path:
    return bundle_dir(bundle_id) / f'{index:04d}.jpg'


def frame_paths(meta: dict) -> list[Path]:
    return [frame_path(meta['id'], index) for index in range(meta['num_frames'])]


def read_meta(bundle_id: str) -> dict | None:
    """The bundle's meta.json, or None when no such bundle exists."""
    if not UUID_RE.match(bundle_id):
        return None
    path = bundle_dir(bundle_id) / META_FILENAME
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def write_bundle(bundle_id: str, frames: list[bytes], meta: dict) -> Path:
    directory = bundle_dir(bundle_id)
    directory.mkdir(parents=True, exist_ok=False)
    for index, data in enumerate(frames):
        frame_path(bundle_id, index).write_bytes(data)
    (directory / META_FILENAME).write_text(json.dumps(meta))
    return directory


def owns_bundle(meta: dict, user) -> bool:
    return meta['user_id'] == user.id or user.role == 'admin'


def load_owned_meta(bundle_id: str, user) -> dict:
    """meta.json of a bundle the user may use: 404 when missing, 403 when it belongs to someone else."""
    meta = read_meta(bundle_id)
    if meta is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f'Video frames bundle {bundle_id!r} not found',
        )
    if not owns_bundle(meta, user):
        log.warning('User %s tried to use video frames bundle %s owned by %s', user.id, bundle_id, meta['user_id'])
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f'Video frames bundle {bundle_id!r} belongs to another user',
        )
    return meta


############################
# Transport
############################


def _server_path(path: Path) -> str:
    if VIDEO_SHARED_PATH is None:
        return str(path)
    return str(VIDEO_SHARED_PATH / path.relative_to(UPLOAD_DIR))


def _data_uri(path: Path, content_type: str) -> str:
    return f'data:{content_type};base64,{base64.b64encode(path.read_bytes()).decode()}'


def _uploaded_file_path(file_id: str) -> Path:
    """Local path of a file uploaded through /api/v1/files (stored as UPLOAD_DIR/<id>_<name>)."""
    matches = sorted(Path(UPLOAD_DIR).glob(f'{file_id}_*')) if UUID_RE.match(file_id) else []
    if not matches:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f'Uploaded video file {file_id!r} not found in the upload directory',
        )
    return matches[0]


def _frames_part(meta: dict) -> dict:
    if VIDEO_FRAMES_TRANSPORT == 'path':
        frames = [_server_path(path) for path in frame_paths(meta)]
    elif VIDEO_FRAMES_TRANSPORT == 'data_uri':
        frames = [_data_uri(path, FRAME_CONTENT_TYPE) for path in frame_paths(meta)]
    else:
        raise ValueError(f'Unsupported VIDEO_FRAMES_TRANSPORT {VIDEO_FRAMES_TRANSPORT!r}')
    return {
        'type': 'video_frames',
        'video_frames': {'frames': frames, 'fps': meta['fps'], 'duration': meta['duration']},
    }


def _file_part(item: dict) -> dict:
    file_id = item.get('id') or item.get('url') or ''
    path = _uploaded_file_path(file_id)
    if VIDEO_FRAMES_TRANSPORT == 'path':
        return {'type': 'video', 'video': _server_path(path)}
    if VIDEO_FRAMES_TRANSPORT == 'data_uri':
        content_type = item.get('content_type') or mimetypes.guess_type(path.name)[0]
        if not content_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f'Cannot determine the content type of video file {file_id!r}',
            )
        return {'type': 'video_url', 'video_url': {'url': _data_uri(path, content_type)}}
    raise ValueError(f'Unsupported VIDEO_FRAMES_TRANSPORT {VIDEO_FRAMES_TRANSPORT!r}')


############################
# Request expansion
############################


def _is_video_item(item: dict) -> bool:
    return item.get('type') == 'video' or (item.get('content_type') or '').startswith('video/')


def _is_image_item(item: dict) -> bool:
    return item.get('type') == 'image' or (item.get('content_type') or '').startswith('image/')


def _extract_text_video_parts(text: str) -> tuple[str, list[dict]]:
    """Split bare video URL tokens out of the text into video / video_url parts."""
    parts = []

    def to_part(match: re.Match) -> str:
        url = match.group(0)
        if url.lower().startswith('file://'):
            parts.append({'type': 'video', 'video': url[len('file://') :]})
        else:
            parts.append({'type': 'video_url', 'video_url': {'url': url}})
        return ''

    return VIDEO_URL_RE.sub(to_part, text).strip(), parts


def _inject_text_video_urls(message: dict) -> None:
    content = message.get('content')
    if isinstance(content, str):
        text, parts = _extract_text_video_parts(content)
        if parts:
            message['content'] = [{'type': 'text', 'text': text}, *parts]
    elif isinstance(content, list):
        new_content = []
        for part in content:
            if isinstance(part, dict) and part.get('type') == 'text':
                text, video_parts = _extract_text_video_parts(part.get('text', ''))
                if video_parts:
                    new_content.append({**part, 'text': text})
                    new_content.extend(video_parts)
                    continue
            new_content.append(part)
        message['content'] = new_content


def max_pixels_for(bundles: list[dict], temporal_patch_size: int) -> int:
    """Pixel budget that makes the server's resize a no-op: max over bundles of G x T x h x w (DESIGN 2)."""
    return max(
        math.ceil(meta['num_frames'] / temporal_patch_size) * temporal_patch_size * meta['width'] * meta['height']
        for meta in bundles
    )


def inject_video_parts(form_data: dict, model: dict, user) -> dict:
    """Expand the video items attached to user messages into LLM content parts.

    A message carrying video gets its complete content list (text, image_url parts as upstream
    builds them, then the video parts) and loses `files`; messages without video are left for the
    caller's own image injection. Bare video URLs in user text become video parts too. When the
    model prices video per request, `video_pixels` is set from the expanded frame bundles.
    """
    bundles = []
    for message in form_data.get('messages') or []:
        if message.get('role') != 'user':
            continue
        files = message.get('files') or []
        video_items = [item for item in files if _is_video_item(item)]
        if video_items:
            content = message.get('content', '')
            parts = [{'type': 'text', 'text': content}] if isinstance(content, str) else list(content)
            parts.extend(
                {'type': 'image_url', 'image_url': {'url': item['url']}}
                for item in files
                if _is_image_item(item) and item.get('url')
            )
            for item in video_items:
                if 'video_frames' in item:
                    meta = load_owned_meta((item['video_frames'] or {}).get('id') or '', user)
                    bundles.append(meta)
                    parts.append(_frames_part(meta))
                else:
                    parts.append(_file_part(item))
            message['content'] = parts
            message.pop('files', None)
        _inject_text_video_urls(message)

    if bundles and 'video_pixels' not in form_data:
        pixels = (model.get('video_input') or {}).get('pixels') or {}
        if pixels.get('per_request_pixels'):
            temporal_patch_size = pixels.get('temporal_patch_size')
            if not temporal_patch_size:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail='Model video_input.pixels.temporal_patch_size is missing; cannot compute video_pixels',
                )
            form_data['video_pixels'] = {'max_pixels': max_pixels_for(bundles, temporal_patch_size)}
    return form_data
