import { WEBUI_API_BASE_URL } from '$lib/constants';

/** mlx-vlm's answer to POST /v1/prefill/calibrate, passed through unchanged. */
export type PrefillCalibration = {
	route_min_tokens: { before: number; after: number };
	applied: boolean;
	saved_to: string | null;
	points: { prompt_tokens: number; mac_seconds: number; windows_seconds: number }[];
	elapsed_seconds: number;
	/** null unless the Mac and Windows timings do not cross within the measured range. */
	note: string | null;
};

/**
 * Measures where prefilling on the Windows worker becomes faster than on the Mac for a model's
 * mlx-vlm server and, with `apply`, makes that the new break-even → POST /api/v1/prefill/calibrate
 * (admin only). Takes 1-2 minutes; chats sent meanwhile wait behind it.
 */
export const calibratePrefill = async (
	token: string,
	modelId: string,
	apply: boolean = true
): Promise<PrefillCalibration> => {
	const res = await fetch(`${WEBUI_API_BASE_URL}/prefill/calibrate`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		},
		body: JSON.stringify({ model_id: modelId, apply })
	});

	if (!res.ok) {
		const detail = (await res.json().catch(() => null))?.detail;
		throw new Error(
			typeof detail === 'string'
				? detail
				: detail
					? JSON.stringify(detail)
					: `HTTP ${res.status} ${res.statusText}`
		);
	}
	return res.json();
};
