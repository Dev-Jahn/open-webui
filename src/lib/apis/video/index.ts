import { WEBUI_API_BASE_URL } from '$lib/constants';
import type { VideoFramesRef } from '$lib/utils/video';

export type VideoFramesMeta = {
	fps: number;
	duration: number;
	width: number;
	height: number;
	num_frames: number;
	name?: string;
	content_type?: string;
};

const errorMessage = (err: any): string =>
	err?.detail ?? err?.message ?? (typeof err === 'string' ? err : 'Request failed');

/** Uploads an ordered set of JPEG frames as one bundle → POST /api/v1/video/frames. */
export const uploadVideoFrames = async (
	token: string,
	frames: Blob[],
	meta: VideoFramesMeta
): Promise<VideoFramesRef> => {
	const data = new FormData();
	frames.forEach((blob, index) => {
		data.append('frames', blob, `${String(index).padStart(4, '0')}.jpg`);
	});
	data.append('meta', JSON.stringify(meta));

	const res = await fetch(`${WEBUI_API_BASE_URL}/video/frames`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			authorization: `Bearer ${token}`
		},
		body: data
	});

	if (!res.ok) {
		const body = await res.json().catch(() => ({ detail: res.statusText }));
		throw new Error(errorMessage(body));
	}
	return res.json();
};

/** Removes a frame bundle → DELETE /api/v1/video/frames/{id}. */
export const deleteVideoFrames = async (token: string, id: string): Promise<boolean> => {
	const res = await fetch(`${WEBUI_API_BASE_URL}/video/frames/${id}`, {
		method: 'DELETE',
		headers: {
			Accept: 'application/json',
			authorization: `Bearer ${token}`
		}
	});

	if (!res.ok) {
		const body = await res.json().catch(() => ({ detail: res.statusText }));
		throw new Error(errorMessage(body));
	}
	return true;
};

/** URL of one frame image of a bundle (auth via the session cookie, like file previews). */
export const videoFrameUrl = (id: string, index: number): string =>
	`${WEBUI_API_BASE_URL}/video/frames/${id}/${index}`;
