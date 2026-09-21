// Client-side planning for video input (frame sampling + smart resize), mirroring the
// mlx-vlm server math so the frames we send are accepted as-is (no server-side resize).
//
// Server contract: GET /v1/models exposes `video_input` per model; the sampling/pixel
// limits below come straight from it (see DESIGN.md §1-2 of the video-frames feature).

export type VideoSampling = {
	fps: number;
	min_frames: number;
	max_frames: number;
	frame_factor: number;
};

export type VideoPixels = {
	min_pixels: number;
	max_pixels_total: number;
	max_pixels_per_frame: number | null;
	patch_size: number;
	temporal_patch_size: number;
	merge_size: number;
	size_factor: number;
	per_request_pixels: boolean;
};

export type VideoInputInfo = {
	supported: boolean;
	sampling: VideoSampling | null;
	pixels: VideoPixels | null;
};

export type VideoInputMode = 'frames' | 'file';

/** Per-model user settings, persisted under `$settings.videoInput[modelId]`. */
export type VideoInputSettings = {
	mode: VideoInputMode;
	maxFrames: number;
	tokensPerFrame: number;
};

export type VideoMeta = { duration: number; width: number; height: number };

export type VideoPlan = {
	n: number; // frames to sample (uniformly)
	fps: number; // effective sampling rate = n / duration (what the server is told)
	width: number; // frame size after smart resize
	height: number;
	tokens: number; // approximate visual tokens for the whole clip
};

/** Shape of a persisted frames-mode bundle reference on a message file item. */
export type VideoFramesRef = {
	id: string;
	num_frames: number;
	width: number;
	height: number;
	fps: number;
	duration: number;
};

export const DEFAULT_MAX_FRAMES = 32;
export const DEFAULT_TOKENS_PER_FRAME = 768; // Qwen-VL video default: 768 × size_factor² pixels
export const TOKENS_PER_FRAME_STEP = 64;
export const TOKENS_PER_FRAME_FLOOR = 64;
export const TOKENS_PER_FRAME_DEFAULT_CAP = 4096;
export const MAX_ASPECT_RATIO = 200;

/**
 * Reads `video_input` from a model object as delivered by GET /api/models. Presets
 * (custom models with `info.base_model_id`) inherit it from their base model when
 * `models` is given. Returns null when the model does not advertise video input.
 */
export const getVideoInputInfo = (
	model: unknown,
	models: unknown[] = []
): VideoInputInfo | null => {
	const m = model as Record<string, any> | null | undefined;
	if (!m) return null;

	const own = m.video_input ?? m.openai?.video_input ?? null;
	if (own && typeof own === 'object' && typeof own.supported === 'boolean') {
		return own as VideoInputInfo;
	}

	const baseId = m.info?.base_model_id;
	if (baseId && baseId !== m.id) {
		const base = models.find((x) => (x as Record<string, any>)?.id === baseId);
		return base ? getVideoInputInfo(base) : null;
	}
	return null;
};

export const supportsVideoInput = (model: unknown, models: unknown[] = []): boolean =>
	getVideoInputInfo(model, models)?.supported === true;

/** A message file item that carries a video (frames mode or original-file mode). */
export const isVideoFile = (file: unknown): boolean => {
	const f = file as Record<string, any> | null | undefined;
	return f?.type === 'video' || String(f?.content_type ?? '').startsWith('video/');
};

const sizeFactor = (info: VideoInputInfo): number => info.pixels?.size_factor ?? 32;

const roundToStep = (value: number, step: number): number => Math.round(value / step) * step;

const clamp = (value: number, min: number, max: number): number =>
	Math.min(Math.max(value, min), max);

/** Upper bound of the "tokens per frame" slider for this model. */
export const tokensPerFrameCap = (info: VideoInputInfo): number => {
	const perFrame = info.pixels?.max_pixels_per_frame;
	if (!perFrame) return TOKENS_PER_FRAME_DEFAULT_CAP;
	const F = sizeFactor(info);
	const cap = Math.floor(perFrame / (F * F) / TOKENS_PER_FRAME_STEP) * TOKENS_PER_FRAME_STEP;
	return Math.max(TOKENS_PER_FRAME_FLOOR, cap);
};

/** [min, max, step] for the "max frames" slider. */
export const maxFramesRange = (info: VideoInputInfo): [number, number, number] => {
	const s = info.sampling;
	return [s?.min_frames ?? 1, s?.max_frames ?? DEFAULT_MAX_FRAMES, s?.frame_factor ?? 1];
};

const clampMaxFrames = (value: number, info: VideoInputInfo): number => {
	const [min, max, step] = maxFramesRange(info);
	const snapped = Math.max(step, roundToStep(value, step));
	return clamp(snapped, min, max);
};

const clampTokensPerFrame = (value: number, info: VideoInputInfo): number => {
	const snapped = Math.max(TOKENS_PER_FRAME_FLOOR, roundToStep(value, TOKENS_PER_FRAME_STEP));
	return clamp(snapped, TOKENS_PER_FRAME_FLOOR, tokensPerFrameCap(info));
};

/** Default settings for a model, derived from its advertised limits. */
export const seedVideoInputSettings = (info: VideoInputInfo): VideoInputSettings => ({
	mode: 'frames',
	maxFrames: clampMaxFrames(DEFAULT_MAX_FRAMES, info),
	tokensPerFrame: clampTokensPerFrame(DEFAULT_TOKENS_PER_FRAME, info)
});

/** Stored settings (possibly partial or stale) normalised into the model's valid ranges. */
export const resolveVideoInputSettings = (
	info: VideoInputInfo,
	stored?: Partial<VideoInputSettings> | null
): VideoInputSettings => {
	const seed = seedVideoInputSettings(info);
	return {
		mode: stored?.mode === 'file' ? 'file' : 'frames',
		maxFrames: clampMaxFrames(stored?.maxFrames ?? seed.maxFrames, info),
		tokensPerFrame: clampTokensPerFrame(stored?.tokensPerFrame ?? seed.tokensPerFrame, info)
	};
};

export type SmartResizeOptions = { factor: number; maxPixels: number; minPixels: number };

/**
 * Qwen-VL smart_resize: snap both sides to multiples of `factor`, then scale down (or up)
 * so the area fits within [minPixels, maxPixels] while keeping the aspect ratio.
 */
export const smartResize = (
	height: number,
	width: number,
	{ factor, maxPixels, minPixels }: SmartResizeOptions
): { height: number; width: number } => {
	if (!(height > 0) || !(width > 0)) {
		throw new Error(`Invalid frame size ${width}×${height}`);
	}
	if (Math.max(height, width) / Math.min(height, width) > MAX_ASPECT_RATIO) {
		throw new Error(
			`Aspect ratio of ${width}×${height} exceeds ${MAX_ASPECT_RATIO}:1, which the model rejects`
		);
	}

	let h = Math.max(factor, roundToStep(height, factor));
	let w = Math.max(factor, roundToStep(width, factor));

	if (h * w > maxPixels) {
		const beta = Math.sqrt((height * width) / maxPixels);
		h = Math.floor(height / beta / factor) * factor;
		w = Math.floor(width / beta / factor) * factor;
	} else if (h * w < minPixels) {
		const beta = Math.sqrt(minPixels / (height * width));
		h = Math.ceil((height * beta) / factor) * factor;
		w = Math.ceil((width * beta) / factor) * factor;
	}

	return { height: h, width: w };
};

/** Number of frames the server will expect for a clip of `duration` seconds. */
export const planFrameCount = (
	duration: number,
	sampling: VideoSampling,
	maxFrames: number,
	temporalPatchSize: number
): number => {
	const T = Math.max(1, temporalPatchSize);
	let n = clamp(Math.floor(duration * sampling.fps), sampling.min_frames, maxFrames);
	n = Math.max(T, n - (n % T));
	return n;
};

/**
 * Turns a probed clip + model limits + user settings into the exact sampling/resize plan.
 * Throws when the model does not publish sampling/pixel limits or the clip is unusable.
 */
export const planVideo = (
	meta: VideoMeta,
	info: VideoInputInfo,
	settings: VideoInputSettings
): VideoPlan => {
	if (!info.supported || !info.sampling || !info.pixels) {
		throw new Error('Model does not publish video sampling limits');
	}
	if (!(meta.duration > 0) || !Number.isFinite(meta.duration)) {
		throw new Error('Video duration is unknown');
	}

	const { sampling, pixels } = info;
	const F = pixels.size_factor;
	const T = pixels.temporal_patch_size;

	const n = planFrameCount(meta.duration, sampling, settings.maxFrames, T);
	const { height, width } = smartResize(meta.height, meta.width, {
		factor: F,
		maxPixels: settings.tokensPerFrame * F * F,
		minPixels: pixels.min_pixels
	});

	const tokens = estimateVideoTokens({ num_frames: n, width, height }, info);

	return { n, fps: n / meta.duration, width, height, tokens };
};

/** Approximate visual tokens for a frame bundle: g·G + 8·G (g = merged patches per frame, G = temporal groups). */
export const estimateVideoTokens = (
	bundle: { num_frames: number; width: number; height: number },
	info: VideoInputInfo
): number => {
	const F = info.pixels?.size_factor ?? 32;
	const T = Math.max(1, info.pixels?.temporal_patch_size ?? 2);
	const g = (bundle.height / F) * (bundle.width / F);
	const G = Math.ceil(bundle.num_frames / T);
	return g * G + G * 8;
};

/** Uniform sample times (seconds) for `n` frames over `duration`: t_i = (i + 0.5) · duration / n. */
export const sampleTimes = (duration: number, n: number): number[] =>
	Array.from({ length: n }, (_, i) => ((i + 0.5) * duration) / n);

/**
 * `video_pixels.max_pixels` the request must carry so the server's own resize is a no-op:
 * the largest G·T·h'·w' over the bundles in the request.
 */
export const maxPixelsFor = (
	bundles: { num_frames: number; width: number; height: number }[],
	temporalPatchSize: number
): number => {
	const T = Math.max(1, temporalPatchSize);
	return bundles.reduce((acc, b) => {
		const G = Math.ceil(b.num_frames / T);
		return Math.max(acc, G * T * b.width * b.height);
	}, 0);
};

export const formatTokenCount = (tokens: number): string =>
	tokens >= 1000 ? `${(tokens / 1000).toFixed(1)}k` : `${Math.round(tokens)}`;

const PROBE_TIMEOUT_MS = 15000;

/** Reads duration and frame size from a video file using a detached <video> element. */
export const probeVideo = (file: File): Promise<VideoMeta> =>
	new Promise((resolve, reject) => {
		const url = URL.createObjectURL(file);
		const video = document.createElement('video');
		video.preload = 'metadata';
		video.muted = true;
		video.playsInline = true;

		const cleanup = () => {
			clearTimeout(timer);
			video.onloadedmetadata = null;
			video.onerror = null;
			video.removeAttribute('src');
			video.load();
			URL.revokeObjectURL(url);
		};
		const fail = (message: string) => {
			cleanup();
			reject(new Error(message));
		};
		const timer = setTimeout(
			() => fail(`Timed out reading metadata of "${file.name}"`),
			PROBE_TIMEOUT_MS
		);

		video.onerror = () =>
			fail(
				`Browser could not decode "${file.name}"${video.error?.message ? `: ${video.error.message}` : ''}`
			);
		video.onloadedmetadata = () => {
			const meta = { duration: video.duration, width: video.videoWidth, height: video.videoHeight };
			cleanup();
			if (!Number.isFinite(meta.duration) || meta.duration <= 0) {
				reject(new Error(`Could not determine the duration of "${file.name}"`));
			} else if (!(meta.width > 0) || !(meta.height > 0)) {
				reject(new Error(`"${file.name}" has no video track`));
			} else {
				resolve(meta);
			}
		};
		video.src = url;
	});
