"""Video input for LLM servers that accept video content parts (mlx-vlm / vllm-mlx).

Frame bundles (uniformly sampled JPEG frames extracted in the browser) live under
``UPLOAD_DIR / 'video_frames' / <id> / {0000.jpg, 0001.jpg, ..., meta.json}`` and are
created by ``routers/video.py``. ``inject_media_parts`` turns the image and video items attached to
user messages into content parts (media first, text last) right before the request is sent;
``put_media_first`` restores that order after upstream prepends its file-context text.

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


def frame_timestamps(meta: dict) -> list[float]:
    """Each frame's time in seconds: as stored at upload, or for a bundle stored before timestamps
    existed, the centres of n equal slots of the clip (where that sampler took its frames)."""
    if 'timestamps' in meta:
        return meta['timestamps']
    count, duration = meta['num_frames'], meta['duration']
    return [(index + 0.5) * duration / count for index in range(count)]


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
    """Split bare video URL tokens out of the text into video / video_url parts (text unchanged when none)."""
    parts = []

    def to_part(match: re.Match) -> str:
        url = match.group(0)
        if url.lower().startswith('file://'):
            parts.append({'type': 'video', 'video': url[len('file://') :]})
        else:
            parts.append({'type': 'video_url', 'video_url': {'url': url}})
        return ''

    remaining = VIDEO_URL_RE.sub(to_part, text)
    return (remaining.strip() if parts else text), parts


def _attachment_parts(files: list[dict], user, bundles: list[tuple[dict, dict]]) -> list[dict]:
    """Content parts of the image and video attachments, in attachment order. Images are selected
    exactly as upstream's own injection selects them; each expanded frame bundle is recorded in
    `bundles` as (meta, its video_frames spec)."""
    parts = []
    for item in files:
        if _is_image_item(item):
            if item.get('url'):
                parts.append({'type': 'image_url', 'image_url': {'url': item['url']}})
        elif _is_video_item(item):
            if 'video_frames' in item:
                meta = load_owned_meta((item['video_frames'] or {}).get('id') or '', user)
                part = _frames_part(meta)
                bundles.append((meta, part['video_frames']))
                parts.append(part)
            else:
                parts.append(_file_part(item))
    return parts


def _is_text_part(part) -> bool:
    return isinstance(part, dict) and part.get('type') == 'text'


def _media_first(parts: list) -> list:
    """Qwen order: the non-text parts, then the text parts, each group in its original order."""
    return [part for part in parts if not _is_text_part(part)] + [part for part in parts if _is_text_part(part)]


def put_media_first(messages: list[dict]) -> list[dict]:
    """Reorder every user message's list content as [media..., text...], each group keeping its order.

    Undoes text that upstream puts in front of the media after `inject_media_parts` ran (the
    `<attached_files>` file context). String content and other roles are untouched. The result
    depends only on the parts and a second call changes nothing, so a turn replayed in a later
    request stays byte-identical (the server's prefix cache keys on exact media and order).
    Rewrites the messages in place and returns the same list."""
    for message in messages:
        if message.get('role') == 'user' and isinstance(message.get('content'), list):
            message['content'] = _media_first(message['content'])
    return messages


def _expand_media(message: dict, user, bundles: list[tuple[dict, dict]]) -> None:
    """Rewrite a user message carrying media as [media..., text...] and drop its `files`.

    Media, in order: the non-text parts already in a list content, the image / video attachments,
    the video URLs typed in the text. As in upstream's own injection, attached images join a string
    content only (a list content already holds the images its sender meant). A message without
    media is left untouched."""
    content = message.get('content', '')
    is_string = isinstance(content, str)
    attachments = [item for item in message.get('files') or [] if is_string or not _is_image_item(item)]
    parts, url_videos = [], []
    for part in [{'type': 'text', 'text': content}] if is_string else content or []:
        if _is_text_part(part):
            text, videos = _extract_text_video_parts(part.get('text', ''))
            parts.append({**part, 'text': text})
            url_videos.extend(videos)
        else:
            parts.append(part)
    parts += [*_attachment_parts(attachments, user, bundles), *url_videos]
    if not all(map(_is_text_part, parts)):
        message['content'] = _media_first(parts)
        message.pop('files', None)


def max_pixels_for(bundles: list[dict], temporal_patch_size: int) -> int:
    """Pixel budget that makes the server's resize a no-op: max over bundles of G x T x h x w (DESIGN 2)."""
    return max(
        math.ceil(meta['num_frames'] / temporal_patch_size) * temporal_patch_size * meta['width'] * meta['height']
        for meta in bundles
    )


def resolve_video_input(model: dict, models: dict) -> dict:
    """The model's video_input. A preset (info.base_model_id) has none of its own: it reads its
    base model's entry in `models` (request.app.state.MODELS)."""
    base_model_id = (model.get('info') or {}).get('base_model_id')
    if not base_model_id:
        return model.get('video_input') or {}
    base_model = models.get(base_model_id)
    if base_model is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'Base model {base_model_id!r} of {model.get("id")!r} is not available; '
            'cannot read its video input settings',
        )
    return base_model.get('video_input') or {}


def _reject_mixed_video_modes(messages: list[dict]) -> None:
    """A request carries one video_pixels budget, so sampled frames and whole videos cannot share it."""
    part_types = {
        part.get('type')
        for message in messages
        if isinstance(message.get('content'), list)
        for part in message['content']
        if isinstance(part, dict)
    }
    if 'video_frames' in part_types and part_types & {'video', 'video_url'}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='This chat mixes sampled video frames with an original video file or URL. '
            'One request has a single video pixel budget, so the two video modes cannot be combined; '
            'start a new chat for the other mode.',
        )


def _apply_video_input(form_data: dict, bundles: list[tuple[dict, dict]], video_input: dict) -> None:
    """The model's video_input on the expanded frame bundles: `sampling.timestamps` adds each
    frame's time, `pixels.per_request_pixels` sets the request's `video_pixels`."""
    if (video_input.get('sampling') or {}).get('timestamps') is True:
        for meta, spec in bundles:
            spec['timestamps'] = frame_timestamps(meta)
    if 'video_pixels' in form_data:
        return
    pixels = video_input.get('pixels') or {}
    if pixels.get('per_request_pixels'):
        temporal_patch_size = pixels.get('temporal_patch_size')
        if not temporal_patch_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Model video_input.pixels.temporal_patch_size is missing; cannot compute video_pixels',
            )
        form_data['video_pixels'] = {'max_pixels': max_pixels_for([meta for meta, _ in bundles], temporal_patch_size)}


def inject_media_parts(form_data: dict, model: dict, user, models: dict) -> dict:
    """Turn the image and video items attached to user messages into content parts, media first.

    Every user message with media (image / video attachments, video URLs typed in the text, or
    non-text parts of a list content) becomes [media..., text...] and loses `files`, so the
    caller's own image injection has nothing left to add; messages without media are untouched.
    Only frame bundles need the model's video_input (a preset reads its base model's entry in
    `models`), so it is resolved only when a bundle is present. A request mixing frame bundles
    with whole videos is rejected.
    """
    user_messages = [message for message in form_data.get('messages') or [] if message.get('role') == 'user']
    bundles = []
    for message in user_messages:
        _expand_media(message, user, bundles)
    _reject_mixed_video_modes(user_messages)
    if bundles:
        _apply_video_input(form_data, bundles, resolve_video_input(model, models))
    return form_data
