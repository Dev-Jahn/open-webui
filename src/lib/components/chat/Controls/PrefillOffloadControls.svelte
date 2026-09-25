<script context="module" lang="ts">
	import { writable } from 'svelte/store';
	import type { PrefillCalibration } from '$lib/apis/prefill';

	// Admin break-even measurements by switch id (the model mlx-vlm measures). Module-level, so a
	// measurement still shows as running, then its result, after the Controls pane is reopened.
	const measuring = writable<Record<string, boolean>>({});
	const results = writable<Record<string, PrefillCalibration>>({});
</script>

<script lang="ts">
	import { getContext, onMount } from 'svelte';
	import type { Writable } from 'svelte/store';
	import type { i18n as i18nType } from 'i18next';
	import { toast } from 'svelte-sonner';

	import Collapsible from '$lib/components/common/Collapsible.svelte';
	import Switch from '$lib/components/common/Switch.svelte';
	import { config, models as allModels, settings, user } from '$lib/stores';
	import { getModels } from '$lib/apis';
	import { getUserSettings, updateUserSettings } from '$lib/apis/users';
	import { calibratePrefill } from '$lib/apis/prefill';

	const i18n = getContext<Writable<i18nType>>('i18n');

	/** Full model objects of the current chat (as passed to Controls). */
	export let models: any[] = [];
	export let buttonClassName = 'w-full';

	// Collapsible infers its `title` prop as `null` from the default value; keep the string untyped.
	let title: any;
	$: title = $i18n.t('Prefill Offload');

	// Section open/close state, persisted like the other chat-control sections.
	const OPEN_KEY = 'chatControls.prefillOffload';
	let open = (localStorage.getItem(OPEN_KEY) ?? 'true') === 'true';
	const onOpenChange = (value: boolean) => localStorage.setItem(OPEN_KEY, String(value));

	// The model whose switch applies, when its `/v1/models` entry has `prefill_offload.auto`: the
	// model itself, or a Workspace preset's base model (the same rule as backend utils/prefill.py).
	const offloadEntryOf = (model: any, all: any[]): any | null => {
		const baseId = model?.info?.base_model_id;
		const entry = baseId ? all.find((m) => m?.id === baseId) : model;
		return entry?.prefill_offload?.auto === true ? entry : null;
	};

	// One row per selected model that can offload; a preset's row is its base model's switch.
	let rows: { model: any; switchId: string; breakEven: number | null }[] = [];
	$: rows = models.flatMap((model) => {
		const entry = offloadEntryOf(model, $allModels);
		const breakEven = entry?.prefill_offload?.break_even_tokens;
		return entry
			? [{ model, switchId: entry.id, breakEven: typeof breakEven === 'number' ? breakEven : null }]
			: [];
	});

	/** The top-level user setting `prefillOffload` = {model id: on} as the server last confirmed it
	 * (a missing entry means on); null until loaded. */
	let stored: Record<string, boolean> | null = null;

	onMount(async () => {
		try {
			stored = (await getUserSettings(localStorage.token))?.prefillOffload ?? {};
		} catch (err) {
			toast.error($i18n.t('Could not load the prefill offload switch: {{error}}', { error: err }));
		}
	});

	// The upstream Switch flips itself on click (it has no controlled mode), so a switch moves
	// before the server confirms; after a failed save, bumping this remounts the switches so they
	// show the confirmed value again.
	let reverts = 0;

	const save = async (switchId: string, on: boolean) => {
		try {
			const fresh = (await getUserSettings(localStorage.token))?.prefillOffload ?? {};
			const saved = await updateUserSettings(localStorage.token, {
				prefillOffload: { ...fresh, [switchId]: on }
			});
			// updateUserSettings resolves to null (instead of throwing) when the request never got an answer.
			if (saved?.prefillOffload?.[switchId] !== on) {
				throw new Error('the server did not confirm the change');
			}
			stored = saved.prefillOffload;
		} catch (err) {
			toast.error(
				$i18n.t('Could not save the prefill offload switch: {{error}}', {
					error: (err as Error)?.message ?? err
				})
			);
			reverts += 1;
		}
	};

	// Saves run one after another, so each merges into the result of the previous one.
	let saving = Promise.resolve();
	const flip = (switchId: string, on: boolean) => (saving = saving.then(() => save(switchId, on)));

	const tokens = (n: number) => n.toLocaleString();
	const seconds = (n: number) => n.toFixed(1);

	// Admin only: mlx-vlm times a short and a long prompt on the Mac and on Windows and applies
	// the crossing as the new break-even; the model list is then reloaded so it shows the new value.
	const measure = async (switchId: string) => {
		measuring.update((m) => ({ ...m, [switchId]: true }));
		try {
			const result = await calibratePrefill(localStorage.token, switchId);
			results.update((r) => ({ ...r, [switchId]: result }));
		} catch (err) {
			toast.error((err as Error)?.message ?? String(err));
			return;
		} finally {
			measuring.update((m) => ({ ...m, [switchId]: false }));
		}
		try {
			allModels.set(
				await getModels(
					localStorage.token,
					$config?.features?.enable_direct_connections
						? ($settings?.directConnections ?? null)
						: null
				)
			);
		} catch (err) {
			toast.error(
				$i18n.t('Could not reload the model list: {{error}}', {
					error: (err as any)?.detail ?? (err as Error)?.message ?? err
				})
			);
		}
	};
</script>

{#if rows.length > 0 && stored !== null}
	<Collapsible
		{title}
		bind:open
		onChange={onOpenChange}
		{buttonClassName}
		chevronClassName="size-2.5"
		chevronStrokeWidth="2"
	>
		<div class="pt-1 pb-1 text-xs flex flex-col gap-1" slot="content">
			{#each rows as { model, switchId, breakEven } (model.id)}
				{@const label = rows.length > 1 ? (model.name ?? model.id) : $i18n.t('Prefill on Windows')}
				<div class="flex w-full items-center justify-between gap-2 py-0.5">
					<div class="self-center text-xs line-clamp-1">{label}</div>
					{#key reverts}
						<Switch
							state={stored[switchId] !== false}
							ariaLabel={label}
							on:change={(e) => flip(switchId, e.detail)}
						/>
					{/key}
				</div>
				{#if $user?.role === 'admin'}
					{@const result = $results[switchId]}
					<div
						class="flex w-full items-center justify-between gap-2 text-gray-500 dark:text-gray-400"
					>
						<div class="line-clamp-1">
							{#if breakEven !== null}
								{$i18n.t('Windows from {{tokens}} tokens', { tokens: tokens(breakEven) })}
							{/if}
						</div>
						<button
							class="shrink-0 rounded-sm px-1.5 transition hover:text-gray-900 dark:hover:text-gray-100 disabled:opacity-50 disabled:pointer-events-none"
							type="button"
							disabled={$measuring[switchId]}
							on:click={() => measure(switchId)}
						>
							{$i18n.t('Measure')}
							<span class="text-gray-400 dark:text-gray-500">{$i18n.t('(1-2 min)')}</span>
						</button>
					</div>
					{#if $measuring[switchId]}
						<div class="text-gray-500 dark:text-gray-400">
							{$i18n.t('Measuring... chats wait until it finishes')}
						</div>
					{:else if result}
						<div class="flex flex-col text-gray-500 dark:text-gray-400">
							<div class="text-gray-700 dark:text-gray-300">
								{$i18n.t('Break-even {{before}} → {{after}} tokens', {
									before: tokens(result.route_min_tokens.before),
									after: tokens(result.route_min_tokens.after)
								})}
							</div>
							{#each result.points as point}
								<div>
									{$i18n.t('{{tokens}} tokens: Mac {{mac}} s / Windows {{windows}} s', {
										tokens: tokens(point.prompt_tokens),
										mac: seconds(point.mac_seconds),
										windows: seconds(point.windows_seconds)
									})}
								</div>
							{/each}
							{#if result.note}
								<div>{result.note}</div>
							{/if}
						</div>
					{/if}
				{/if}
			{/each}
			<div class="text-xs text-gray-500 dark:text-gray-400">
				{$i18n.t(
					'When on, long prompts are prefilled on the Windows PC only when that is faster and the PC is free; otherwise on this Mac.'
				)}
			</div>
		</div>
	</Collapsible>
{/if}
