<script lang="ts">
	// One muted line under the message input while the model list is empty because an OpenAI
	// connection is unreachable: how to start mlx-vlm. The text comes from the backend
	// (MLX_VLM_OFFLINE_HINT), so it has no fixed words to translate here.
	import { models } from '$lib/stores';
	import { getConnectionState } from '$lib/apis/prefill';

	let hint: string | null = null;

	const check = async () => {
		hint = null;
		try {
			hint = (await getConnectionState(localStorage.token)).hint;
		} catch (error) {
			console.error('Checking the OpenAI connections failed:', error);
		}
	};

	// Asks each time the list becomes empty; a list that fills hides the line.
	$: empty = $models.length === 0;
	$: if (empty) check();
</script>

{#if empty && hint}
	<div class="px-3 text-center text-xs text-gray-500 dark:text-gray-400" role="status">
		{hint}
	</div>
{/if}
