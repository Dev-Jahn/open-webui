// Frame extraction in the browser: <video> seek + <canvas> draw, one frame at a time.
// WebCodecs (VideoDecoder) is deliberately not used — it needs a container demuxer.
//
// Positions follow the Qwen standard (evenly spaced from the first frame to the last) and every
// frame carries the presentation time of the frame actually drawn: requestVideoFrameCallback's
// `mediaTime`. Measured in Chrome 153 (2026-09-24) on paused, detached elements with blob: URLs:
// - one callback per seek, about one display frame after it, sometimes just before `seeked`; it has
//   to be registered BEFORE currentTime is set (registered after `seeked` it misses the frame);
// - a seek to another position inside the same source frame still calls back, with the same
//   mediaTime (that is how duplicates show); a seek to the current position calls back never;
// - a hidden tab still gets `seeked` but no callback until it is visible again;
// - mediaTime is rounded to whole microseconds, so a target equal to a frame's reported start can
//   fall just before its true start and draw the previous frame (29.97 fps: the last frame is
//   missed; n = frame count: a third of the frames duplicated) — hence SEEK_NUDGE_S;
// - currentTime reports the requested position, not the frame's own time.
// Other engines (Playwright WebKit 26.4 and Firefox 150, 2026-09-24):
// - WebKit reports frame starts like Chrome, in the first rendering update after `seeked`, but
//   never calls back for a seek that stays inside the frame already shown; such a seek is
//   recognised by drawing the frame and comparing it with the previous capture;
// - Firefox reports the requested position (to the microsecond) instead of the frame's start, so
//   repeated frames cannot be told apart; extraction stops with an error there.

import { frameTargets, type VideoPlan } from './index';

export type ExtractFramesOptions = {
	onProgress?: (done: number, total: number) => void;
	signal?: AbortSignal;
};

/** One JPEG per sample target, in temporal order, with the presentation time of each. */
export type CapturedFrames = { frames: Blob[]; times: number[] };

export const JPEG_QUALITY = 0.85;
const LOAD_TIMEOUT_MS = 20000;
const SEEK_TIMEOUT_MS = 10000;
/** Seek this far past each target, so a frame starting exactly at the target is the one drawn. */
const SEEK_NUDGE_S = 1e-4;
/** Never seek exactly to `duration`: some decoders never fire `seeked` for it. */
const END_MARGIN_S = 0.001;
/** Rendering updates to wait after `seeked` for a frame report before checking for an unchanged frame. */
const UNCHANGED_FRAME_GRACE = 3;
/** A reported time this close to the requested position is that position echoed back (Firefox). */
const ECHO_TOLERANCE_S = 2e-6;

const abortError = () => new DOMException('Frame extraction was cancelled', 'AbortError');

const decodeError = (video: HTMLVideoElement) =>
	new Error(
		`Browser could not decode the video${video.error?.message ? `: ${video.error.message}` : ''}`
	);

const throwIfAborted = (signal?: AbortSignal) => {
	if (signal?.aborted) throw abortError();
};

/** Resolves on `loadeddata`, rejects on the element's `error` event, abort or timeout. */
const waitForLoad = (video: HTMLVideoElement, signal?: AbortSignal): Promise<void> =>
	new Promise((resolve, reject) => {
		const done = (fn: () => void) => () => {
			clearTimeout(timer);
			video.removeEventListener('loadeddata', onLoaded);
			video.removeEventListener('error', onError);
			signal?.removeEventListener('abort', onAbort);
			fn();
		};
		const onLoaded = done(resolve);
		const onError = done(() => reject(decodeError(video)));
		const onAbort = done(() => reject(abortError()));
		const timer = setTimeout(
			done(() => reject(new Error('Timed out while loading the video'))),
			LOAD_TIMEOUT_MS
		);

		video.addEventListener('loadeddata', onLoaded, { once: true });
		video.addEventListener('error', onError, { once: true });
		signal?.addEventListener('abort', onAbort, { once: true });
	});

const hasFrameCallbacks = (video: HTMLVideoElement): boolean =>
	typeof video.requestVideoFrameCallback === 'function';

/** The frame captured last: its presentation time, and whether the element still shows it. */
type LastCapture = { time: number; isShown: () => boolean };

/**
 * Seeks to `position` and resolves, once the seek finished and the frame there was presented, with
 * that frame's presentation time. Without requestVideoFrameCallback (older browsers) the time is
 * currentTime, i.e. the position itself. When `seeked` came but no frame was reported within a few
 * rendering updates (WebKit, seek inside the frame already shown), the frame still shown is `last`
 * and keeps its time if it looks exactly like that capture. The timeout only runs while the page is
 * visible, as hidden pages get no frame callbacks; it rejects, never guessing a time.
 */
const seekFrame = (
	video: HTMLVideoElement,
	position: number,
	describe: () => string,
	last: LastCapture | null,
	signal?: AbortSignal
): Promise<number> =>
	new Promise((resolve, reject) => {
		const frameCallbacks = hasFrameCallbacks(video);
		let seeked = false;
		let presentedAt: number | null = null;
		let callbackId: number | null = null;
		let animationId: number | null = null;
		let newFrameUnreported = false;
		let timer: ReturnType<typeof setTimeout>;

		const finish = (fn: () => void) => {
			clearTimeout(timer);
			if (callbackId !== null) video.cancelVideoFrameCallback(callbackId);
			if (animationId !== null) cancelAnimationFrame(animationId);
			document.removeEventListener('visibilitychange', onVisible);
			video.removeEventListener('seeked', onSeeked);
			video.removeEventListener('error', onError);
			signal?.removeEventListener('abort', onAbort);
			fn();
		};
		const settle = () => {
			const time = presentedAt;
			if (seeked && time !== null) finish(() => resolve(time));
		};
		const checkUnchanged = (updatesLeft: number) => {
			animationId = requestAnimationFrame(() => {
				animationId = null;
				if (updatesLeft > 1) checkUnchanged(updatesLeft - 1);
				else if (last?.isShown()) finish(() => resolve(last.time));
				else newFrameUnreported = true; // keep waiting for its report until the timeout
			});
		};
		const onSeeked = () => {
			seeked = true;
			if (!frameCallbacks) presentedAt = video.currentTime;
			if (presentedAt !== null) settle();
			else if (last) checkUnchanged(UNCHANGED_FRAME_GRACE);
		};
		const onError = () => finish(() => reject(decodeError(video)));
		const onAbort = () => finish(() => reject(abortError()));
		const onTimeout = () => {
			if (document.hidden) document.addEventListener('visibilitychange', onVisible);
			else
				finish(() =>
					reject(
						new Error(
							newFrameUnreported
								? `The browser showed a new frame without reporting its time while ${describe()}`
								: `Timed out while ${describe()}`
						)
					)
				);
		};
		const onVisible = () => {
			if (document.hidden) return;
			document.removeEventListener('visibilitychange', onVisible);
			timer = setTimeout(onTimeout, SEEK_TIMEOUT_MS);
		};

		video.addEventListener('seeked', onSeeked, { once: true });
		video.addEventListener('error', onError, { once: true });
		signal?.addEventListener('abort', onAbort, { once: true });
		if (frameCallbacks) {
			callbackId = video.requestVideoFrameCallback((_, metadata) => {
				presentedAt = metadata.mediaTime;
				settle();
			});
		}
		timer = setTimeout(onTimeout, SEEK_TIMEOUT_MS);
		video.currentTime = position;
	});

const canvasToJpeg = (canvas: HTMLCanvasElement): Promise<Blob> =>
	new Promise((resolve, reject) => {
		canvas.toBlob(
			(blob) =>
				blob ? resolve(blob) : reject(new Error('Canvas could not encode the frame as JPEG')),
			'image/jpeg',
			JPEG_QUALITY
		);
	});

/**
 * Captures `plan.n` frames of `file` at positions evenly spaced from its first frame to its last,
 * resized to `plan.width`×`plan.height`. Frames are returned as captured: two targets on the same
 * source frame give two equal times (see selectDistinctFrames).
 */
export const extractFrames = async (
	file: File,
	plan: VideoPlan,
	{ onProgress, signal }: ExtractFramesOptions = {}
): Promise<CapturedFrames> => {
	throwIfAborted(signal);

	const url = URL.createObjectURL(file);
	const video = document.createElement('video');
	video.preload = 'auto';
	video.muted = true;
	video.playsInline = true;

	const canvas = document.createElement('canvas');
	canvas.width = plan.width;
	canvas.height = plan.height;
	const ctx = canvas.getContext('2d');
	if (!ctx) {
		URL.revokeObjectURL(url);
		throw new Error('Canvas 2D context is not available');
	}
	ctx.imageSmoothingEnabled = true;
	ctx.imageSmoothingQuality = 'high';

	try {
		const loaded = waitForLoad(video, signal);
		video.src = url;
		await loaded;

		const duration = video.duration;
		if (!Number.isFinite(duration) || duration <= 0) {
			throw new Error('Video duration is unknown');
		}
		const end = Math.max(0, duration - END_MARGIN_S);

		// The last frame's own time bounds the targets.
		let presented = await seekFrame(video, end, () => 'finding the last frame', null, signal);
		if (hasFrameCallbacks(video) && Math.abs(presented - end) < ECHO_TOLERANCE_S) {
			throw new Error(
				'This browser reports the requested seek position instead of the time of the frame it shows (Firefox does), so repeated frames cannot be detected. Use Chrome or Safari, or send the video as "Original file".'
			);
		}
		const targets = frameTargets(presented, plan.n);

		// Whether the element still shows the frame captured last, which the canvas holds.
		const showsLastCapture = () => {
			const captured = ctx.getImageData(0, 0, plan.width, plan.height).data;
			ctx.drawImage(video, 0, 0, plan.width, plan.height);
			const shown = ctx.getImageData(0, 0, plan.width, plan.height).data;
			return shown.every((value, index) => value === captured[index]);
		};

		const frames: Blob[] = [];
		const times: number[] = [];
		for (const [i, t] of targets.entries()) {
			throwIfAborted(signal);
			const position = Math.min(t + SEEK_NUDGE_S, end);
			// Seeking to the current position presents nothing new: the frame on screen is the one.
			if (position !== video.currentTime) {
				presented = await seekFrame(
					video,
					position,
					() => `seeking to ${t.toFixed(2)}s (frame ${i + 1}/${plan.n})`,
					i > 0 ? { time: presented, isShown: showsLastCapture } : null,
					signal
				);
			}

			ctx.drawImage(video, 0, 0, plan.width, plan.height);
			frames.push(await canvasToJpeg(canvas));
			times.push(presented);
			onProgress?.(frames.length, plan.n);
		}
		throwIfAborted(signal);
		return { frames, times };
	} finally {
		video.pause();
		video.removeAttribute('src');
		video.load();
		URL.revokeObjectURL(url);
	}
};
