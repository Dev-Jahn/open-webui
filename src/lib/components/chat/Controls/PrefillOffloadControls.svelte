<script lang="ts">
	import { getContext, onMount } from 'svelte';
	import type { Writable } from 'svelte/store';
	import type { i18n as i18nType } from 'i18next';
	import { toast } from 'svelte-sonner';

	import Collapsible from '$lib/components/common/Collapsible.svelte';
	import Switch from '$lib/components/common/Switch.svelte';
	import { models as allModels } from '$lib/stores';
	import { getUserSettings, updateUserSettings } from '$lib/apis/users';

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
	const switchIdOf = (model: any, all: any[]): string | null => {
		const baseId = model?.info?.base_model_id;
		const entry = baseId ? all.find((m) => m?.id === baseId) : model;
		return entry?.prefill_offload?.auto === true ? entry.id : null;
	};

	// One row per selected model that can offload; a preset's row is its base model's switch.
	let rows: { model: any; switchId: string }[] = [];
	$: rows = models.flatMap((model) => {
		const switchId = switchIdOf(model, $allModels);
		return switchId ? [{ model, switchId }] : [];
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
			{#each rows as { model, switchId } (model.id)}
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
			{/each}
			<div class="text-xs text-gray-500 dark:text-gray-400">
				{$i18n.t(
					'When on, long prompts are prefilled on the Windows PC only when that is faster and the PC is free; otherwise on this Mac.'
				)}
			</div>
		</div>
	</Collapsible>
{/if}
