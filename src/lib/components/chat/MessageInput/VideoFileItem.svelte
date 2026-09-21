<script lang="ts">
	import { createEventDispatcher, getContext } from 'svelte';
	import type { Writable } from 'svelte/store';
	import type { i18n as i18nType } from 'i18next';

	import { WEBUI_API_BASE_URL } from '$lib/constants';
	import { models, settings } from '$lib/stores';
	import { formatFileSize } from '$lib/utils';
	import { videoFrameUrl } from '$lib/apis/video';
	import {
		estimateVideoTokens,
		formatTokenCount,
		getVideoInputInfo,
		type VideoFramesRef
	} from '$lib/utils/video';

	import Spinner from '$lib/components/common/Spinner.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import Play from '$lib/components/icons/Play.svelte';
	import XMark from '$lib/components/icons/XMark.svelte';
	import VideoFramesPlayer from './VideoFramesPlayer.svelte';

	const i18n = getContext<Writable<i18nType>>('i18n');
	const dispatch = createEventDispatcher();

	export let file: any;
	/** Models the message will go to; a warning is shown for those without video input. */
	export let selectedModelIds: string[] = [];
	export let dismissible = false;
	/** Message view: poster / player instead of the compact input tile. */
	export let large = false;

	let frames: VideoFramesRef | null = null;
	$: frames = file?.video_frames ?? null;

	/** Flipbook of the sampled frames (what the model saw), opened from the poster. */
	let showPlayer = false;

	$: uploading = file?.status === 'uploading';
	$: posterUrl = frames ? videoFrameUrl(frames.id, 0) : null;
	$: videoUrl =
		!frames && file?.url
			? file.url.startsWith('data') || file.url.startsWith('http')
				? file.url
				: `${WEBUI_API_BASE_URL}/files/${file.url}/content`
			: null;

	$: unsupportedModels = selectedModelIds.filter(
		(id) =>
			!getVideoInputInfo(
				$models.find((m) => m.id === id),
				$models
			)?.supported
	);

	$: info = getVideoInputInfo(
		$models.find((m) => m.id === selectedModelIds[0]),
		$models
	);
	$: tokens = frames && info?.supported ? estimateVideoTokens(frames, info) : null;

	let hint = '';
	$: if (uploading) {
		const p = file?.progress;
		hint = !p?.total
			? $i18n.t('Reading video…')
			: p.uploading
				? $i18n.t('Uploading frames…')
				: $i18n.t('Extracting {{done}}/{{total}}…', { done: p.done, total: p.total });
	} else if (frames) {
		hint = [
			$i18n.t('{{count}} frames', { count: frames.num_frames }),
			`${frames.width}×${frames.height}`,
			...(tokens !== null ? [`≈ ${formatTokenCount(tokens)} ${$i18n.t('tokens')}`] : [])
		].join(' · ');
	} else {
		hint = [$i18n.t('Original file'), ...(file?.size ? [formatFileSize(file.size)] : [])].join(
			' · '
		);
	}
</script>

{#if large}
	{#if frames}
		<div class="group relative inline-block">
			<button
				type="button"
				aria-label={$i18n.t('Play sampled frames')}
				class="block cursor-pointer"
				on:click={() => (showPlayer = true)}
			>
				<img src={posterUrl} alt={file?.name ?? ''} class="max-h-96 rounded-lg" />
				<div
					class="absolute inset-0 flex items-center justify-center opacity-0 transition group-hover:opacity-100 group-focus-visible:opacity-100"
				>
					<div class="rounded-full bg-black/60 p-3 text-white">
						<Play className="size-6" strokeWidth="2" />
					</div>
				</div>
			</button>
			<div
				class="pointer-events-none absolute bottom-1.5 left-1.5 rounded-md bg-black/60 px-1.5 py-0.5 text-xs text-white"
			>
				{$i18n.t('{{count}} frames', { count: frames.num_frames })} · {frames.fps.toFixed(1)} fps
			</div>
		</div>
	{:else if videoUrl}
		<!-- svelte-ignore a11y-media-has-caption -->
		<video src={videoUrl} class="max-h-96 rounded-lg" controls muted playsinline preload="metadata"
		></video>
	{/if}
{:else}
	<div
		class="relative group flex items-center gap-2 max-w-64 rounded-2xl p-1.5 pr-3 bg-white dark:bg-gray-850 border border-gray-50/30 dark:border-gray-800/30"
	>
		<div
			class="relative size-10 shrink-0 overflow-hidden rounded-xl bg-black/20 dark:bg-white/10 flex items-center justify-center text-white"
		>
			{#if uploading}
				<Spinner className="size-4" />
			{:else if posterUrl}
				<button
					type="button"
					aria-label={$i18n.t('Play sampled frames')}
					class="size-full cursor-pointer"
					on:click={() => (showPlayer = true)}
				>
					<img src={posterUrl} alt="" class="size-full object-cover" />
				</button>
			{:else if videoUrl}
				<!-- svelte-ignore a11y-media-has-caption -->
				<video src={videoUrl} class="size-full object-cover" muted playsinline preload="metadata"
				></video>
			{/if}

			{#if unsupportedModels.length > 0}
				<Tooltip
					className="absolute top-0.5 left-0.5"
					content={$i18n.t('Video input is not supported by: {{models}}', {
						models: unsupportedModels.join(', ')
					})}
				>
					<svg
						xmlns="http://www.w3.org/2000/svg"
						viewBox="0 0 24 24"
						fill="currentColor"
						aria-hidden="true"
						class="size-4 fill-yellow-300"
					>
						<path
							fill-rule="evenodd"
							d="M9.401 3.003c1.155-2 4.043-2 5.197 0l7.355 12.748c1.154 2-.29 4.5-2.599 4.5H4.645c-2.309 0-3.752-2.5-2.598-4.5L9.4 3.003ZM12 8.25a.75.75 0 0 1 .75.75v3.75a.75.75 0 0 1-1.5 0V9a.75.75 0 0 1 .75-.75Zm0 8.25a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5Z"
							clip-rule="evenodd"
						/>
					</svg>
				</Tooltip>
			{/if}
		</div>

		<div class="flex min-w-0 flex-col justify-center -space-y-0.5">
			<div class="text-sm font-normal dark:text-gray-100 line-clamp-1">{file?.name ?? ''}</div>
			<div class="text-xs text-gray-500 dark:text-gray-400 line-clamp-1">{hint}</div>
		</div>

		{#if dismissible}
			<div class=" absolute -top-1 -right-1">
				<button
					aria-label={$i18n.t('Remove file')}
					class=" bg-white text-black border border-gray-50 rounded-full {($settings?.highContrastMode ??
					false)
						? ''
						: 'hover-reveal transition'}"
					type="button"
					on:click|stopPropagation={() => {
						dispatch('dismiss');
					}}
				>
					<XMark className={'size-4'} />
				</button>
			</div>
		{/if}
	</div>
{/if}

{#if frames}
	<VideoFramesPlayer bind:show={showPlayer} {frames} name={file?.name ?? ''} />
{/if}
