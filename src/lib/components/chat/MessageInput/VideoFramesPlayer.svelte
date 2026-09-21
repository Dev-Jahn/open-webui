<script lang="ts">
	// Flipbook of the sampled + resized frames a message actually carried (not the original
	// file): steps through /video/frames/{id}/{i} at the bundle's own rate, looping.
	import { getContext, onDestroy } from 'svelte';
	import type { Writable } from 'svelte/store';
	import type { i18n as i18nType } from 'i18next';

	import { videoFrameUrl } from '$lib/apis/video';
	import type { VideoFramesRef } from '$lib/utils/video';

	import Modal from '$lib/components/common/Modal.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import ChevronLeft from '$lib/components/icons/ChevronLeft.svelte';
	import ChevronRight from '$lib/components/icons/ChevronRight.svelte';
	import Play from '$lib/components/icons/Play.svelte';
	import XMark from '$lib/components/icons/XMark.svelte';

	const i18n = getContext<Writable<i18nType>>('i18n');

	export let show = false;
	export let frames: VideoFramesRef;
	export let name = '';

	let index = 0;
	let playing = false;
	let loaded: boolean[] = [];
	let firstFrameFailed = false;
	let timer: ReturnType<typeof setInterval> | null = null;

	$: count = frames.num_frames;
	$: urls = Array.from({ length: count }, (_, i) => videoFrameUrl(frames.id, i));
	$: ready = loaded[0] === true;
	// Frames were sampled at the centre of each slot: t_i = (i + 0.5) / fps.
	$: caption = [
		$i18n.t('frame {{current}} / {{total}}', { current: index + 1, total: count }),
		`t = ${((index + 0.5) / frames.fps).toFixed(2)} s`,
		`${frames.width}×${frames.height}`,
		`${frames.fps.toFixed(1)} fps`
	].join(' · ');

	const stop = () => {
		if (timer) clearInterval(timer);
		timer = null;
		playing = false;
	};
	const start = () => {
		stop();
		timer = setInterval(() => (index = (index + 1) % frames.num_frames), 1000 / frames.fps);
		playing = true;
	};
	const step = (delta: number) => {
		stop();
		index = (index + delta + frames.num_frames) % frames.num_frames;
	};
	const seek = (e: Event) => {
		stop();
		index = Number((e.currentTarget as HTMLInputElement).value);
	};

	// Open → rewind and autoplay; close (Esc / backdrop / button, via bind:show) → stop.
	$: if (show) {
		index = 0;
		loaded = [];
		firstFrameFailed = false;
		start();
	} else {
		stop();
	}

	// Escape is handled by Modal. Handled keys are consumed so the focused button (the focus
	// trap starts on "Close") or the scrubber does not act on them a second time; Space is
	// consumed on keyup as well because buttons activate on keyup.
	const onKeyDown = (e: KeyboardEvent) => {
		if (!show) return;
		if (e.key === ' ') playing ? stop() : start();
		else if (e.key === 'ArrowLeft') step(-1);
		else if (e.key === 'ArrowRight') step(1);
		else return;
		e.preventDefault();
	};
	const onKeyUp = (e: KeyboardEvent) => {
		if (show && e.key === ' ') e.preventDefault();
	};

	onDestroy(stop);
</script>

<svelte:window on:keydown={onKeyDown} on:keyup={onKeyUp} />

<Modal bind:show size="lg" className="bg-black text-white rounded-2xl">
	<div class="flex flex-col gap-3 p-4">
		<div class="flex items-center justify-between gap-2">
			<div class="min-w-0 truncate text-sm font-medium">{name}</div>
			<button
				type="button"
				aria-label={$i18n.t('Close')}
				class="shrink-0 rounded-full p-1 transition hover:bg-white/10"
				on:click={() => (show = false)}
			>
				<XMark className="size-5" />
			</button>
		</div>

		<!-- Natural frame size, scaled down to fit the modal width or 70% of the viewport height.
		     All frames are stacked in this box so stepping is a visibility toggle: no refetch, no flicker. -->
		<div
			class="relative mx-auto bg-black"
			style="width: min(100%, {frames.width}px, calc(70vh * {frames.width} / {frames.height})); aspect-ratio: {frames.width} / {frames.height};"
		>
			{#each urls as url, i (url)}
				<img
					src={url}
					alt={i === index ? `${name} – ${caption}` : ''}
					decoding="sync"
					class="absolute inset-0 size-full object-contain {i === index ? '' : 'invisible'}"
					on:load={() => (loaded[i] = true)}
					on:error={() => (firstFrameFailed = firstFrameFailed || i === 0)}
				/>
			{/each}

			{#if firstFrameFailed}
				<div class="absolute inset-0 flex items-center justify-center p-4 text-center text-sm">
					{$i18n.t('The sampled frames of this video are no longer available')}
				</div>
			{:else if !ready}
				<div class="absolute inset-0 flex items-center justify-center">
					<Spinner className="size-6" />
				</div>
			{/if}
		</div>

		<div class="flex items-center gap-1">
			<button
				type="button"
				aria-label={$i18n.t('Previous frame')}
				class="rounded-full p-1.5 transition hover:bg-white/10"
				on:click={() => step(-1)}
			>
				<ChevronLeft className="size-4" strokeWidth="2" />
			</button>
			<button
				type="button"
				aria-label={playing ? $i18n.t('Pause') : $i18n.t('Play')}
				class="rounded-full p-1.5 transition hover:bg-white/10"
				on:click={() => (playing ? stop() : start())}
			>
				{#if playing}
					<svg
						xmlns="http://www.w3.org/2000/svg"
						viewBox="0 0 24 24"
						fill="currentColor"
						aria-hidden="true"
						class="size-4"
					>
						<path
							d="M7 5.25A.75.75 0 0 1 7.75 4.5h2.5a.75.75 0 0 1 .75.75v13.5a.75.75 0 0 1-.75.75h-2.5a.75.75 0 0 1-.75-.75V5.25Zm6 0a.75.75 0 0 1 .75-.75h2.5a.75.75 0 0 1 .75.75v13.5a.75.75 0 0 1-.75.75h-2.5a.75.75 0 0 1-.75-.75V5.25Z"
						/>
					</svg>
				{:else}
					<Play className="size-4" strokeWidth="2" />
				{/if}
			</button>
			<button
				type="button"
				aria-label={$i18n.t('Next frame')}
				class="rounded-full p-1.5 transition hover:bg-white/10"
				on:click={() => step(1)}
			>
				<ChevronRight className="size-4" strokeWidth="2" />
			</button>

			<input
				type="range"
				aria-label={$i18n.t('Frame')}
				class="ml-2 h-2 flex-1 cursor-pointer appearance-none rounded-lg bg-gray-700"
				min="0"
				max={count - 1}
				step="1"
				value={index}
				on:input={seek}
			/>
		</div>

		<div class="text-center text-xs tabular-nums text-gray-400">{caption}</div>
	</div>
</Modal>
