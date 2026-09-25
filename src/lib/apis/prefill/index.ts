import { WEBUI_API_BASE_URL } from '$lib/constants';

/** One measured prompt length: prefill seconds locally and on the prefill offload worker. */
export type PrefillCalibrationPoint = {
	prompt_tokens: number;
	local_seconds: number;
	offload_seconds: number;
};

/** mlx-vlm's answer to POST /v1/prefill/calibrate, passed through unchanged except `points`. */
export type PrefillCalibration = {
	route_min_tokens: { before: number; after: number };
	applied: boolean;
	saved_to: string | null;
	points: PrefillCalibrationPoint[];
	elapsed_seconds: number;
	/** null unless the local and offload timings do not cross within the measured range. */
	note: string | null;
};

/** mlx-vlm's calibration point, which must carry both timings (mlx-vlm c586c03b or later). */
const calibrationPoint = (point: any): PrefillCalibrationPoint => {
	const { prompt_tokens, local_seconds, offload_seconds } = point ?? {};
	if (typeof local_seconds !== 'number' || typeof offload_seconds !== 'number') {
		throw new Error(`mlx-vlm sent a calibration point without timings: ${JSON.stringify(point)}`);
	}
	return { prompt_tokens, local_seconds, offload_seconds };
};

/**
 * Measures where prefilling on the prefill offload worker becomes faster than locally for a
 * model's mlx-vlm server and, with `apply`, makes that the new break-even
 * → POST /api/v1/prefill/calibrate (admin only). Takes 1-2 minutes; chats sent meanwhile wait
 * behind it.
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
	const result = await res.json();
	return { ...result, points: result.points.map(calibrationPoint) };
};

/** Whether the OpenAI connections are reachable; `hint` says how to start mlx-vlm when one is not. */
export type ConnectionState = { reachable: boolean; hint: string | null };

/**
 * Checks that every enabled OpenAI connection accepts a connection (1-2 s at most)
 * → GET /api/v1/prefill/connection (any verified user).
 */
export const getConnectionState = async (token: string): Promise<ConnectionState> => {
	const res = await fetch(`${WEBUI_API_BASE_URL}/prefill/connection`, {
		headers: { Accept: 'application/json', authorization: `Bearer ${token}` }
	});
	if (!res.ok) {
		throw new Error(`HTTP ${res.status} ${res.statusText}`);
	}
	return res.json();
};
