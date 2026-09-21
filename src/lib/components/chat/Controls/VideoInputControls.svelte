<script lang="ts">
	import { getContext, onDestroy } from 'svelte';
	import type { Writable } from 'svelte/store';
	import type { i18n as i18nType } from 'i18next';

	import Collapsible from '$lib/components/common/Collapsible.svelte';
	import { settings } from '$lib/stores';
	import { updateUserSettings } from '$lib/apis/users';
	import {
		getVideoInputInfo,
		maxFramesRange,
		resolveVideoInputSettings,
		tokensPerFrameCap,
		SAMPLING_FPS_MAX,
		SAMPLING_FPS_MIN,
		SAMPLING_FPS_STEP,
		TOKENS_PER_FRAME_FLOOR,
		TOKENS_PER_FRAME_STEP,
		type VideoInputInfo,
		type VideoInputSettings
	} from '$lib/utils/video';

	const i18n = getContext<Writable<i18nType>>('i18n');

	/** Full model objects of the current chat (as passed to Controls). */
	export let models: any[] = [];
	export let buttonClassName = 'w-full';

	// Collapsible infers its `title` prop as `null` from the default value; keep the string untyped.
	let title: any;
	$: title = $i18n.t('Video Input');

	// Section open/close state, persisted like the other chat-control sections.
	const OPEN_KEY = 'chatControls.videoInput';
	let open = (localStorage.getItem(OPEN_KEY) ?? 'true') === 'true';
	const onOpenChange = (value: boolean) => localStorage.setItem(OPEN_KEY, String(value));

	let videoModels: { model: any; info: VideoInputInfo }[] = [];
	$: videoModels = models.flatMap((model) => {
		const info = getVideoInputInfo(model, models);
		return info?.supported ? [{ model, info }] : [];
	});

	const PERSIST_DELAY_MS = 600;
	let persistTimer: ReturnType<typeof setTimeout> | null = null;

	const persist = async () => {
		persistTimer = null;
		await updateUserSettings(localStorage.token, { ui: $settings });
	};

	const update = (modelId: string, info: VideoInputInfo, patch: Partial<VideoInputSettings>) => {
		const current = resolveVideoInputSettings(info, $settings?.videoInput?.[modelId]);
		const next = resolveVideoInputSettings(info, { ...current, ...patch });
		settings.set({
			...$settings,
			videoInput: { ...($settings?.videoInput ?? {}), [modelId]: next }
		});

		if (persistTimer) clearTimeout(persistTimer);
		persistTimer = setTimeout(persist, PERSIST_DELAY_MS);
	};

	const numberOf = (e: Event) => Number((e.currentTarget as HTMLInputElement).value);

	onDestroy(() => {
		if (persistTimer) {
			clearTimeout(persistTimer);
			persist();
		}
	});
</script>

<Collapsible
	{title}
	bind:open
	onChange={onOpenChange}
	{buttonClassName}
	chevronClassName="size-2.5"
	chevronStrokeWidth="2"
>
	<div class="pt-1 pb-1 text-xs flex flex-col gap-2" slot="content">
		{#each videoModels as { model, info } (model.id)}
			{@const value = resolveVideoInputSettings(info, $settings?.videoInput?.[model.id])}
			{@const [minFrames, maxFrames, frameStep] = maxFramesRange(info)}
			{@const tokenCap = tokensPerFrameCap(info)}
			<div class="flex flex-col gap-1">
				{#if videoModels.length > 1}
					<div class="text-xs font-medium text-gray-500 dark:text-gray-400 line-clamp-1">
						{model.name ?? model.id}
					</div>
				{/if}

				<div class="flex w-full items-center justify-between py-0.5">
					<div class="self-center text-xs">{$i18n.t('Send video as')}</div>
					<button
						class="p-1 px-3 text-xs flex rounded-sm transition shrink-0 outline-hidden"
						type="button"
						on:click={() =>
							update(model.id, info, { mode: value.mode === 'frames' ? 'file' : 'frames' })}
					>
						<span class="ml-2 self-center">
							{value.mode === 'frames' ? $i18n.t('Sampled frames') : $i18n.t('Original file')}
						</span>
					</button>
				</div>

				{#if value.mode === 'frames'}
					<div class="py-0.5 w-full">
						<div class="text-xs">{$i18n.t('Max Frames')}</div>
						<div class="flex mt-0.5 space-x-2">
							<div class="flex-1">
								<input
									type="range"
									aria-label={$i18n.t('Max Frames')}
									min={minFrames}
									max={maxFrames}
									step={frameStep}
									value={value.maxFrames}
									on:input={(e) => update(model.id, info, { maxFrames: numberOf(e) })}
									class="w-full h-2 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
								/>
							</div>
							<div>
								<input
									type="number"
									aria-label={$i18n.t('Max Frames')}
									class="bg-transparent text-center w-14"
									min={minFrames}
									max={maxFrames}
									step={frameStep}
									value={value.maxFrames}
									on:change={(e) => update(model.id, info, { maxFrames: numberOf(e) })}
								/>
							</div>
						</div>
					</div>

					<div class="py-0.5 w-full">
						<div class="text-xs">{$i18n.t('Sampling FPS')}</div>
						<div class="flex mt-0.5 space-x-2">
							<div class="flex-1">
								<input
									type="range"
									aria-label={$i18n.t('Sampling FPS')}
									min={SAMPLING_FPS_MIN}
									max={SAMPLING_FPS_MAX}
									step={SAMPLING_FPS_STEP}
									value={value.fps}
									on:input={(e) => update(model.id, info, { fps: numberOf(e) })}
									class="w-full h-2 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
								/>
							</div>
							<div>
								<input
									type="number"
									aria-label={$i18n.t('Sampling FPS')}
									class="bg-transparent text-center w-14"
									min={SAMPLING_FPS_MIN}
									max={SAMPLING_FPS_MAX}
									step={SAMPLING_FPS_STEP}
									value={value.fps}
									on:change={(e) => update(model.id, info, { fps: numberOf(e) })}
								/>
							</div>
						</div>
					</div>

					<div class="py-0.5 w-full">
						<div class="text-xs">{$i18n.t('Tokens per Frame')}</div>
						<div class="flex mt-0.5 space-x-2">
							<div class="flex-1">
								<input
									type="range"
									aria-label={$i18n.t('Tokens per Frame')}
									min={TOKENS_PER_FRAME_FLOOR}
									max={tokenCap}
									step={TOKENS_PER_FRAME_STEP}
									value={value.tokensPerFrame}
									on:input={(e) => update(model.id, info, { tokensPerFrame: numberOf(e) })}
									class="w-full h-2 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
								/>
							</div>
							<div>
								<input
									type="number"
									aria-label={$i18n.t('Tokens per Frame')}
									class="bg-transparent text-center w-14"
									min={TOKENS_PER_FRAME_FLOOR}
									max={tokenCap}
									step={TOKENS_PER_FRAME_STEP}
									value={value.tokensPerFrame}
									on:change={(e) => update(model.id, info, { tokensPerFrame: numberOf(e) })}
								/>
							</div>
						</div>
					</div>
				{/if}
			</div>
		{/each}
	</div>
</Collapsible>
