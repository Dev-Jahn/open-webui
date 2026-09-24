import asyncio
import json
import logging
import shutil
import time
import uuid
from io import BytesIO

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from open_webui.constants import ERROR_MESSAGES
from open_webui.utils.auth import get_verified_user
from open_webui.utils.video import (
    FRAME_CONTENT_TYPE,
    bundle_dir,
    frame_path,
    owns_bundle,
    read_meta,
    write_bundle,
)
from PIL import Image
from pydantic import BaseModel, Field, ValidationError

log = logging.getLogger(__name__)

router = APIRouter()

# How far (seconds) a frame's time may pass the reported duration: the last frame's
# presentation time can sit slightly beyond a container's rounded duration.
TIMESTAMP_SLACK = 0.05


############################
# Video frame bundles
# Frames sampled from a video in the browser; see utils/video.py for storage and expansion.
############################


class VideoFramesMetaForm(BaseModel):
    fps: float = Field(gt=0)
    duration: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    num_frames: int = Field(ge=1)
    name: str | None = None
    content_type: str | None = None


class VideoFramesResponse(BaseModel):
    id: str
    num_frames: int
    width: int
    height: int
    fps: float
    duration: float
    timestamps: list[float] | None = None


def _invalid_timestamps(reason: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f'Invalid timestamps: {reason}')


def _parse_timestamps(raw: str, num_frames: int, duration: float) -> list[float]:
    """The optional `timestamps` form field: a JSON array with each frame's time in seconds."""
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as e:
        raise _invalid_timestamps(f'not JSON: {e}')
    if not isinstance(values, list) or any(isinstance(t, bool) or not isinstance(t, (int, float)) for t in values):
        raise _invalid_timestamps('expected a JSON array of numbers')
    if len(values) != num_frames:
        raise _invalid_timestamps(f'{len(values)} values for {num_frames} frames')
    # Chained comparisons also reject NaN and infinity.
    if not all(0 <= t <= duration + TIMESTAMP_SLACK for t in values):
        raise _invalid_timestamps(f'every value must be a finite number of seconds from 0 to the duration {duration}')
    if any(later < earlier for earlier, later in zip(values, values[1:])):
        raise _invalid_timestamps('values must be non-decreasing')
    return [float(t) for t in values]


def _check_frame(data: bytes, index: int, width: int, height: int) -> None:
    try:
        with Image.open(BytesIO(data)) as image:
            image.load()
            image_format, size = image.format, image.size
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'Frame {index} is not a decodable image: {e}',
        )
    if image_format != 'JPEG':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'Frame {index} must be a JPEG image, got {image_format}',
        )
    if size != (width, height):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'Frame {index} is {size[0]}x{size[1]}, expected {width}x{height}',
        )


def _get_owned_meta_or_404(id: str, user) -> dict:
    meta = read_meta(id)
    if meta is None or not owns_bundle(meta, user):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )
    return meta


@router.post('/frames', response_model=VideoFramesResponse, response_model_exclude_none=True)
async def upload_video_frames(
    frames: list[UploadFile] = File([]),
    meta: str = Form(...),
    timestamps: str | None = Form(None),
    user=Depends(get_verified_user),
):
    try:
        form = VideoFramesMetaForm.model_validate_json(meta)
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'Invalid meta: {e}',
        )

    if form.num_frames != len(frames):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'meta.num_frames is {form.num_frames} but {len(frames)} frames were uploaded',
        )
    frame_times = None if timestamps is None else _parse_timestamps(timestamps, form.num_frames, form.duration)

    frame_bytes = [await frame.read() for frame in frames]

    def check_and_write() -> dict:
        for index, data in enumerate(frame_bytes):
            _check_frame(data, index, form.width, form.height)
        bundle_meta = {
            'id': str(uuid.uuid4()),
            'user_id': user.id,
            'created_at': int(time.time()),
            **form.model_dump(),
            **({'timestamps': frame_times} if frame_times is not None else {}),
        }
        write_bundle(bundle_meta['id'], frame_bytes, bundle_meta)
        return bundle_meta

    bundle_meta = await asyncio.to_thread(check_and_write)
    log.info('Stored video frames bundle %s (%d frames) for user %s', bundle_meta['id'], len(frames), user.id)
    return VideoFramesResponse(**bundle_meta)


@router.get('/frames/{id}/{index}')
async def get_video_frame(id: str, index: int, user=Depends(get_verified_user)):
    meta = _get_owned_meta_or_404(id, user)
    if not 0 <= index < meta['num_frames']:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )
    return FileResponse(frame_path(id, index), media_type=FRAME_CONTENT_TYPE)


@router.delete('/frames/{id}')
async def delete_video_frames(id: str, user=Depends(get_verified_user)):
    _get_owned_meta_or_404(id, user)
    await asyncio.to_thread(shutil.rmtree, bundle_dir(id))
    return {'message': 'Video frames deleted successfully'}
