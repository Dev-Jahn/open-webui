// Frame extraction in the browser: <video> seek + <canvas> draw, one frame at a time.
// WebCodecs (VideoDecoder) is deliberately not used in v1 — it needs a container demuxer.
// The signature is kept so a WebCodecs path can be added later without touching callers.

import { sampleTimes, type VideoPlan } from './index';

export type ExtractFramesOptions = {
	onProgress?: (done: number, total: number) => void;
	signal?: AbortSignal;
};

export const JPEG_QUALITY = 0.85;
const LOAD_TIMEOUT_MS = 20000;
const SEEK_TIMEOUT_MS = 10000;

const abortError = () => new DOMException('Frame extraction was cancelled', 'AbortError');

const throwIfAborted = (signal?: AbortSignal) => {
	if (signal?.aborted) throw abortError();
};

/** Resolves on `event`, rejects on the element's `error` event, abort or timeout. */
const waitForEvent = (
	video: HTMLVideoElement,
	event: 'loadeddata' | 'seeked',
	timeoutMs: number,
	describe: () => string,
	signal?: AbortSignal
): Promise<void> =>
	new Promise((resolve, reject) => {
		const done = (fn: () => void) => () => {
			clearTimeout(timer);
			video.removeEventListener(event, onEvent);
			video.removeEventListener('error', onError);
			signal?.removeEventListener('abort', onAbort);
			fn();
		};
		const onEvent = done(resolve);
		const onError = done(() =>
			reject(
				new Error(
					`Browser could not decode the video${video.error?.message ? `: ${video.error.message}` : ''}`
				)
			)
		);
		const onAbort = done(() => reject(abortError()));
		const timer = setTimeout(
			done(() => reject(new Error(`Timed out while ${describe()}`))),
			timeoutMs
		);

		video.addEventListener(event, onEvent, { once: true });
		video.addEventListener('error', onError, { once: true });
		signal?.addEventListener('abort', onAbort, { once: true });
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
 * Samples `plan.n` frames uniformly from `file`, resized to `plan.width`×`plan.height`,
 * and returns them as JPEG blobs in temporal order.
 */
export const extractFrames = async (
	file: File,
	plan: VideoPlan,
	{ onProgress, signal }: ExtractFramesOptions = {}
): Promise<Blob[]> => {
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
		const loaded = waitForEvent(
			video,
			'loadeddata',
			LOAD_TIMEOUT_MS,
			() => 'loading the video',
			signal
		);
		video.src = url;
		await loaded;

		const duration = video.duration;
		if (!Number.isFinite(duration) || duration <= 0) {
			throw new Error('Video duration is unknown');
		}

		const frames: Blob[] = [];
		const times = sampleTimes(duration, plan.n);
		for (const [i, t] of times.entries()) {
			throwIfAborted(signal);
			const seeked = waitForEvent(
				video,
				'seeked',
				SEEK_TIMEOUT_MS,
				() => `seeking to ${t.toFixed(2)}s (frame ${i + 1}/${plan.n})`,
				signal
			);
			// Never seek exactly to `duration`: some decoders never fire `seeked` for it.
			video.currentTime = Math.min(t, Math.max(0, duration - 0.001));
			await seeked;

			ctx.drawImage(video, 0, 0, plan.width, plan.height);
			frames.push(await canvasToJpeg(canvas));
			onProgress?.(frames.length, plan.n);
		}
		throwIfAborted(signal);
		return frames;
	} finally {
		video.pause();
		video.removeAttribute('src');
		video.load();
		URL.revokeObjectURL(url);
	}
};
