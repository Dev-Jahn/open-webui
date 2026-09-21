// Drop-time pipeline for video attachments in the chat input:
//   probe → plan → push item (status 'uploading') → extract frames (progress) → upload bundle → 'uploaded'
// and re-extraction when the per-model settings or the model selection change.
// Original File objects live here (keyed by itemId), never on the file item that gets persisted.

import { toast } from 'svelte-sonner';
import { get } from 'svelte/store';
import { v4 as uuidv4 } from 'uuid';

import i18n from '$lib/i18n';
import { deleteVideoFrames, uploadVideoFrames } from '$lib/apis/video';
import {
	getVideoInputInfo,
	isVideoFile,
	planVideo,
	probeVideo,
	resolveVideoInputSettings,
	type VideoInputInfo,
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
	pending: VideoPlan | null;
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

const isPlanned = (
	plan: VideoPlan,
	ref: { num_frames: number; width: number; height: number } | null | undefined
): boolean =>
	!!ref && ref.num_frames === plan.n && ref.width === plan.width && ref.height === plan.height;

const runExtraction = async (item: any, plan: VideoPlan, ctx: VideoFilesContext) => {
	const entry = entries.get(item.itemId);
	if (!entry) return;

	entry.controller?.abort();
	const controller = new AbortController();
	entry.controller = controller;
	entry.pending = plan;

	item.status = 'uploading';
	item.progress = { done: 0, total: plan.n, uploading: false };
	touch(ctx);

	try {
		const frames = await extractFrames(entry.file, plan, {
			signal: controller.signal,
			onProgress: (done, total) => {
				item.progress = { done, total, uploading: false };
				touch(ctx);
			}
		});

		item.progress = { done: plan.n, total: plan.n, uploading: true };
		touch(ctx);

		const bundle = await uploadVideoFrames(ctx.token, frames, {
			fps: plan.fps,
			duration: entry.meta.duration,
			width: plan.width,
			height: plan.height,
			num_frames: plan.n,
			name: entry.file.name,
			content_type: entry.file.type
		});

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
		entry.pending = null;

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
		entry.pending = null;
		fail(item, e, ctx);
	}
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

	entries.set(item.itemId, { file, meta, controller: null, pending: null });
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

		const current = entry.pending
			? { num_frames: entry.pending.n, width: entry.pending.width, height: entry.pending.height }
			: item.video_frames;
		if (isPlanned(plan, current)) continue;

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
