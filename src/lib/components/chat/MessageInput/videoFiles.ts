// Drop-time pipeline for video attachments in the chat input:
//   probe → plan → push item (status 'uploading') → extract frames (progress) → upload bundle → 'uploaded'
// and re-extraction when the per-model settings or the model selection change.
// Original File objects live here (keyed by itemId), never on the file item that gets persisted.

import { toast } from 'svelte-sonner';
import { get } from 'svelte/store';
import { v4 as uuidv4 } from 'uuid';

import i18n from '$lib/i18n';
import { createMessagesList } from '$lib/utils';
import { deleteVideoFrames, uploadVideoFrames } from '$lib/apis/video';
import {
	getVideoInputInfo,
	isVideoFile,
	planVideo,
	probeVideo,
	resolveVideoInputSettings,
	selectDistinctFrames,
	videoFileMode,
	type VideoInputInfo,
	type VideoInputMode,
	type VideoInputSettings,
	type VideoMeta,
	type VideoPlan
} from '$lib/utils/video';
import { extractFrames } from '$lib/utils/video/extract';

export type VideoFilesContext = {
	getFiles: () => any[];
	setFiles: (files: any[]) => void;
	onUpdate: (file: any) => void;
	models: unknown[];
	selectedModelIds: string[];
	videoSettings: Record<string, Partial<VideoInputSettings>> | undefined;
	/** The chat's message tree: the next message continues the branch ending at `currentId`. */
	history: { currentId: string | null; messages: Record<string, any> };
	temporaryChat: boolean;
	token: string;
	/** "Original file" mode: upload the video as a regular file (process=false) with type 'video'. */
	uploadOriginal: (file: File) => Promise<unknown>;
};

const t = (text: string, params?: Record<string, unknown>): string => get(i18n).t(text, params);

type Entry = {
	file: File;
	meta: VideoMeta;
	controller: AbortController | null;
	/** Plan of the extraction in flight or of the uploaded bundle (which may hold fewer frames). */
	plan: VideoPlan | null;
};

const RESYNC_DELAY_MS = 400;

const entries = new Map<string, Entry>();
let resyncTimer: ReturnType<typeof setTimeout> | null = null;

type Target = { info: VideoInputInfo; settings: VideoInputSettings };

/** Limits + settings of the primary (first) selected model; null when any selected model lacks video input. */
const targetFor = (ctx: VideoFilesContext): Target | null => {
	const ids = ctx.selectedModelIds;
	if (ids.length === 0) return null;

	const infos = ids.map((id) =>
		getVideoInputInfo(
			ctx.models.find((m) => (m as { id?: string })?.id === id),
			ctx.models
		)
	);
	if (infos.some((info) => !info?.supported)) return null;

	const info = infos[0] as VideoInputInfo;
	return { info, settings: resolveVideoInputSettings(info, ctx.videoSettings?.[ids[0]]) };
};

const touch = (ctx: VideoFilesContext) => ctx.setFiles(ctx.getFiles());

const discardBundle = async (token: string, id: string) => {
	try {
		await deleteVideoFrames(token, id);
	} catch (e) {
		console.warn(`Could not delete frame bundle ${id}`, e);
	}
};

const fail = (item: any, e: unknown, ctx: VideoFilesContext) => {
	const message = e instanceof Error ? e.message : String(e);
	item.status = 'error';
	item.error = message;
	delete item.progress;
	entries.delete(item.itemId);

	toast.error(t('Failed to process video: {{error}}', { error: message }));
	// Same as uploadFileHandler: a failed attachment leaves the input instead of blocking it.
	ctx.setFiles(ctx.getFiles().filter((f) => f?.itemId !== item.itemId));
	ctx.onUpdate(item);
};

const samePlan = (a: VideoPlan, b: VideoPlan | null): boolean =>
	!!b && a.n === b.n && a.width === b.width && a.height === b.height;

const runExtraction = async (item: any, plan: VideoPlan, ctx: VideoFilesContext) => {
	const entry = entries.get(item.itemId);
	if (!entry) return;

	entry.controller?.abort();
	const controller = new AbortController();
	entry.controller = controller;
	entry.plan = plan;

	item.status = 'uploading';
	item.progress = { done: 0, total: plan.n, uploading: false };
	touch(ctx);

	try {
		const captured = await extractFrames(entry.file, plan, {
			signal: controller.signal,
			onProgress: (done, total) => {
				item.progress = { done, total, uploading: false };
				touch(ctx);
			}
		});
		// The real frames decide the count (and so the rate the server is told).
		const kept = selectDistinctFrames(captured.times, plan.temporalPatchSize, plan.minFrames);

		item.progress = { done: plan.n, total: plan.n, uploading: true };
		touch(ctx);

		const bundle = await uploadVideoFrames(
			ctx.token,
			kept.map((i) => captured.frames[i]),
			{
				fps: kept.length / entry.meta.duration,
				duration: entry.meta.duration,
				width: plan.width,
				height: plan.height,
				num_frames: kept.length,
				name: entry.file.name,
				content_type: entry.file.type
			},
			kept.map((i) => captured.times[i])
		);

		if (controller.signal.aborted) {
			await discardBundle(ctx.token, bundle.id);
			return;
		}

		const previous: string | undefined = item.video_frames?.id;
		item.video_frames = bundle;
		item.id = bundle.id;
		item.status = 'uploaded';
		item.error = '';
		delete item.progress;
		entry.controller = null;

		touch(ctx);
		ctx.onUpdate(item);

		if (previous && previous !== bundle.id) {
			await discardBundle(ctx.token, previous);
		}
		// Sent or dismissed while extracting: nothing left to re-plan for this item.
		if (!ctx.getFiles().some((f) => f?.itemId === item.itemId)) {
			entries.delete(item.itemId);
		}
	} catch (e) {
		if (controller.signal.aborted) return; // superseded by a newer plan or dismissed
		entry.controller = null;
		fail(item, e, ctx);
	}
};

const modeLabel = (mode: VideoInputMode): string =>
	mode === 'frames' ? t('Sampled frames') : t('Original file');

/** The backend's VIDEO_URL_RE (utils/video.py): a video URL typed in user text, sent as a whole video. */
const VIDEO_URL_RE = /(?<!\S)(?:https?:\/\/|file:\/\/)\S+\.(?:mp4|webm|mov|mkv|avi)(?!\S)/i;

/**
 * Modes of the videos the next request carries: those of the current branch, which the backend
 * replays (attached videos, and typed video URLs as 'file'), and the input's pending files.
 */
const chatVideoModes = (ctx: VideoFilesContext): Set<VideoInputMode> => {
	const branch: { role?: string; content?: unknown; files?: unknown[] }[] = createMessagesList(
		ctx.history,
		ctx.history.currentId
	);
	const files = [...branch.flatMap((message) => message.files ?? []), ...ctx.getFiles()];
	const modes = files.map(videoFileMode);
	if (branch.some((m) => m.role === 'user' && VIDEO_URL_RE.test(String(m.content ?? '')))) {
		modes.push('file');
	}
	return new Set(modes.filter((mode): mode is VideoInputMode => mode !== null));
};

/**
 * One request carries one video pixel budget (`video_pixels`), sized for the frame bundles, so an
 * original-file video next to a bundle would be decoded at the bundle's budget. A chat therefore
 * keeps to one mode; true when `mode` may be added.
 */
const acceptsVideoMode = (mode: VideoInputMode, ctx: VideoFilesContext): boolean => {
	const other: VideoInputMode = mode === 'frames' ? 'file' : 'frames';
	if (!chatVideoModes(ctx).has(other)) return true;

	toast.error(
		t(
			'This chat already has a video sent as "{{existing}}", and one chat cannot mix "{{existing}}" and "{{requested}}". Change "Send video as" in Controls or start a new chat.',
			{ existing: modeLabel(other), requested: modeLabel(mode) }
		)
	);
	return false;
};

/** Handles a dropped/picked video file; rejects loudly when video input is not possible. */
export const addVideoFile = async (file: File, ctx: VideoFilesContext): Promise<void> => {
	if (ctx.temporaryChat) {
		toast.error(t('Video input is not supported in temporary chats'));
		return;
	}
	const target = targetFor(ctx);
	if (!target) {
		toast.error(t('Selected model(s) do not support video inputs'));
		return;
	}
	if (!acceptsVideoMode(target.settings.mode, ctx)) return;

	if (target.settings.mode === 'file') {
		await ctx.uploadOriginal(file);
		return;
	}

	const item: any = {
		type: 'video',
		name: file.name,
		size: file.size,
		content_type: file.type,
		status: 'uploading',
		error: '',
		id: null,
		itemId: uuidv4(),
		video_frames: null, // marks the frames mode from the start (see videoFileMode)
		progress: { done: 0, total: 0, uploading: false }
	};
	ctx.setFiles([...ctx.getFiles(), item]);

	let meta: VideoMeta;
	let plan: VideoPlan;
	try {
		meta = await probeVideo(file);
		plan = planVideo(meta, target.info, target.settings);
	} catch (e) {
		fail(item, e, ctx);
		return;
	}

	entries.set(item.itemId, { file, meta, controller: null, plan: null });
	await runExtraction(item, plan, ctx);
};

/** Pre-send check (Chat.svelte): a video-bearing message going to a model without video input. */
export const warnIfVideoUnsupported = (
	model: unknown,
	models: unknown[],
	messages: { files?: unknown[] }[]
): void => {
	const m = model as { id?: string; name?: string } | null;
	if (!m || getVideoInputInfo(model, models)?.supported) return;
	if (!messages.some((message) => (message.files ?? []).some(isVideoFile))) return;

	toast.error(
		t('Model {{modelName}} does not support video inputs', { modelName: m.name ?? m.id })
	);
};

/** Called when a video item is removed from the input: cancels work and drops its bundle. */
export const forgetVideoFile = (item: any, token: string): void => {
	const entry = entries.get(item?.itemId);
	entry?.controller?.abort();
	entries.delete(item?.itemId);

	const bundleId: string | undefined = item?.video_frames?.id;
	if (bundleId) void discardBundle(token, bundleId);
};

const resync = (ctx: VideoFilesContext) => {
	const files = ctx.getFiles();

	for (const [itemId, entry] of entries) {
		if (!entry.controller && !files.some((f) => f?.itemId === itemId)) {
			entries.delete(itemId);
		}
	}

	const target = targetFor(ctx);
	if (!target) return;

	for (const item of files) {
		if (item?.type !== 'video') continue;
		const entry = entries.get(item.itemId);
		if (!entry) continue;

		let plan: VideoPlan;
		try {
			plan = planVideo(entry.meta, target.info, target.settings);
		} catch (e) {
			fail(item, e, ctx);
			continue;
		}
		if (samePlan(plan, entry.plan)) continue;

		void runExtraction(item, plan, ctx);
	}
};

/**
 * Re-plans the frames-mode videos in the input against the current model selection and
 * settings (debounced); items whose plan changed are re-extracted and re-uploaded.
 */
export const resyncVideoFiles = (ctx: VideoFilesContext): void => {
	if (resyncTimer) clearTimeout(resyncTimer);
	resyncTimer = setTimeout(() => {
		resyncTimer = null;
		resync(ctx);
	}, RESYNC_DELAY_MS);
};
