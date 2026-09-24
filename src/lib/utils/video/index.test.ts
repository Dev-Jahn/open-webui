import { describe, expect, it } from 'vitest';
import {
	estimateVideoTokens,
	formatMarkerSeconds,
	frameTargets,
	groupMarkerTimes,
	maxPixelsFor,
	planVideo,
	resolveVideoInputSettings,
	selectDistinctFrames,
	seedVideoInputSettings,
	slotCentreTimes,
	smartResize,
	videoFileMode,
	type VideoInputInfo
} from './index';

// Mirrors what mlx-vlm advertises for Qwen3-VL (DESIGN.md §1).
const info: VideoInputInfo = {
	supported: true,
	sampling: { fps: 2.0, min_frames: 4, max_frames: 768, frame_factor: 2 },
	pixels: {
		min_pixels: 4096,
		max_pixels_total: 25165824,
		max_pixels_per_frame: null,
		patch_size: 16,
		temporal_patch_size: 2,
		merge_size: 2,
		size_factor: 32,
		per_request_pixels: true
	}
};

describe('planVideo', () => {
	it('matches the server-verified example: 640×352, 4 s, fps 2 → 8 frames, 912 tokens', () => {
		const plan = planVideo(
			{ duration: 4, width: 640, height: 352 },
			info,
			resolveVideoInputSettings(info, { maxFrames: 32, tokensPerFrame: 768 })
		);
		expect(plan.n).toBe(8);
		expect(plan.width).toBe(640);
		expect(plan.height).toBe(352);
		expect(
			estimateVideoTokens({ num_frames: 8, width: 640, height: 352, fps: 2, duration: 4 }, info)
		).toBe(912);
	});

	it('scales the frame count with the Sampling FPS setting', () => {
		const plan = (fps: number, maxFrames = 32) =>
			planVideo(
				{ duration: 4, width: 640, height: 352 },
				info,
				resolveVideoInputSettings(info, { maxFrames, fps, tokensPerFrame: 768 })
			);

		expect(plan(4).n).toBe(16);

		// floor(4 s × 0.5) = 2 → min_frames 4 → still 4 after temporal-patch rounding
		expect(plan(0.5).n).toBe(4);

		// Max Frames still caps: 4 s × 8 fps = 32 → 8
		expect(plan(8, 8).n).toBe(8);
	});

	it('clamps to the max-frames slider and keeps n a multiple of the temporal patch', () => {
		const plan = planVideo(
			{ duration: 60, width: 640, height: 352 },
			info,
			resolveVideoInputSettings(info, { maxFrames: 30, tokensPerFrame: 768 })
		);
		expect(plan.n).toBe(30);
	});

	it('never goes below min_frames for very short clips', () => {
		const plan = planVideo(
			{ duration: 0.4, width: 640, height: 352 },
			info,
			seedVideoInputSettings(info)
		);
		expect(plan.n).toBe(4);
	});

	it('rejects a model without sampling limits', () => {
		expect(() =>
			planVideo(
				{ duration: 4, width: 640, height: 352 },
				{ supported: true, sampling: null, pixels: null },
				seedVideoInputSettings(info)
			)
		).toThrow();
	});
});

describe('smartResize', () => {
	it('fits 1920×1080 into 768 tokens: area ≤ 786,432, multiples of 32, aspect preserved', () => {
		const F = 32;
		const maxPixels = 768 * F * F;
		const { height, width } = smartResize(1080, 1920, { factor: F, maxPixels, minPixels: 4096 });

		expect(height * width).toBeLessThanOrEqual(786432);
		expect(height % F).toBe(0);
		expect(width % F).toBe(0);

		const ratio = 1920 / 1080;
		const tolerance = F / height + F / width; // one snapping step on each side
		expect(Math.abs(width / height - ratio) / ratio).toBeLessThanOrEqual(tolerance);
	});

	it('leaves an already-aligned small frame untouched', () => {
		expect(smartResize(352, 640, { factor: 32, maxPixels: 786432, minPixels: 4096 })).toEqual({
			height: 352,
			width: 640
		});
	});

	it('scales tiny frames up to min_pixels', () => {
		const { height, width } = smartResize(20, 40, {
			factor: 32,
			maxPixels: 786432,
			minPixels: 4096
		});
		expect(height * width).toBeGreaterThanOrEqual(4096);
	});

	it('rejects absurd aspect ratios', () => {
		expect(() =>
			smartResize(10, 4000, { factor: 32, maxPixels: 786432, minPixels: 4096 })
		).toThrow();
	});
});

describe('frame sampling', () => {
	it('spaces targets evenly from the first frame to the last (Qwen linspace)', () => {
		expect(frameTargets(9, 4)).toEqual([0, 3, 6, 9]);
		expect(frameTargets(9.966667, 2)).toEqual([0, 9.966667]);
		expect(frameTargets(9.966667, 1)).toEqual([0]);
		const targets = frameTargets(3599.966667, 768);
		expect(targets[0]).toBe(0);
		expect(targets[767]).toBe(3599.966667);
	});

	it('keeps the slot-centre times of older bundles', () => {
		expect(slotCentreTimes(4, 8)).toEqual([0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75]);
	});

	it('drops targets that landed on the same source frame and keeps the count even', () => {
		// 7 captures, 5 distinct → the last distinct one goes so the count stays a multiple of 2
		expect(selectDistinctFrames([0, 0, 0.2, 0.4, 0.4, 0.6, 0.8], 2, 4)).toEqual([0, 2, 3, 5]);
		expect(selectDistinctFrames([0, 0.5, 1, 1.5], 2, 4)).toEqual([0, 1, 2, 3]);
		expect(selectDistinctFrames([0, 0.5, 0.5, 1, 1.5], 2, 4)).toEqual([0, 1, 3, 4]);
	});

	it('fails instead of sending fewer than min_frames', () => {
		expect(() => selectDistinctFrames([0, 0, 0, 0.2, 0.2, 0.4], 2, 4)).toThrow(/only 3 distinct/);
		expect(() => selectDistinctFrames([0, 0, 0, 0], 2, 1)).toThrow(/only 1 distinct/);
	});
});

describe('token estimate', () => {
	it("formats marker seconds like Python's '{:.1f}' (ties to even on the exact binary value)", () => {
		// Expected strings produced by CPython 3 '{:.1f}'.format(x)
		const python: [number, string][] = [
			[0.25, '0.2'],
			[0.75, '0.8'],
			[1.25, '1.2'],
			[2.25, '2.2'],
			[3.25, '3.2'],
			[1.75, '1.8'],
			[0.35, '0.3'],
			[0.45, '0.5'],
			[0.05, '0.1'],
			[0.15, '0.1'],
			[2.5, '2.5'],
			[9.95, '9.9'],
			[9.96, '10.0'],
			[99.95, '100.0'],
			[0, '0.0']
		];
		for (const [x, s] of python) expect(formatMarkerSeconds(x), String(x)).toBe(s);
	});

	it('marks each group with the mean of its first and last frame; a short last group reuses its frame', () => {
		expect(groupMarkerTimes([0, 1, 2, 3], 2)).toEqual([0.5, 2.5]);
		expect(groupMarkerTimes([0, 1, 2], 2)).toEqual([0.5, 2]);
	});

	const hour = { num_frames: 768, width: 960, height: 512, fps: 768 / 3600, duration: 3600 };

	it('counts marker digits: 1 h, 768 frames, 960×512 → 188,425 (flat 8 per group gave 187,392)', () => {
		// Server without timestamps: markers from index / rate
		expect(estimateVideoTokens(hour, info)).toBe(188425);
		// Server with timestamps, bundle from before real times (slot centres)
		const withTimes = { ...info, sampling: { ...info.sampling!, timestamps: true } };
		expect(estimateVideoTokens(hour, withTimes)).toBe(188425);
		// Server with timestamps, linspace frames of a 30 fps clip (last frame at 3599.9667 s)
		const linspace = { ...hour, timestamps: frameTargets(3600 - 1 / 30, 768) };
		expect(estimateVideoTokens(linspace, withTimes)).toBe(188425);
	});

	it('picks the digit count from the formatted time, not the raw value', () => {
		const withTimes = { ...info, sampling: { ...info.sampling!, timestamps: true } };
		const pair = (a: number, b: number) =>
			estimateVideoTokens(
				{ num_frames: 2, width: 32, height: 32, fps: 1, duration: 11, timestamps: [a, b] },
				withTimes
			);
		expect(pair(9.9, 10.0)).toBe(1 + 6 + 2); // mean 9.95 → "9.9"
		expect(pair(9.9, 10.1)).toBe(1 + 7 + 2); // mean 10.0 → "10.0"
		expect(pair(99.9, 100.1)).toBe(1 + 8 + 2);
		expect(pair(9999.9, 10000.1)).toBe(1 + 10 + 2); // one token per digit, also past 4 digits
	});
});

describe('helpers', () => {
	it('tells frames-mode and original-file video items apart', () => {
		expect(videoFileMode({ type: 'video', video_frames: null })).toBe('frames');
		expect(videoFileMode({ type: 'video', video_frames: { id: 'x' } })).toBe('frames');
		expect(videoFileMode({ type: 'video', id: 'f' })).toBe('file');
		expect(videoFileMode({ type: 'file', content_type: 'video/mp4' })).toBe('file');
		expect(videoFileMode({ type: 'image' })).toBeNull();
	});

	it('computes video_pixels.max_pixels as the largest G·T·h·w', () => {
		expect(maxPixelsFor([{ num_frames: 8, width: 640, height: 352 }], 2)).toBe(4 * 2 * 640 * 352);
		expect(
			maxPixelsFor(
				[
					{ num_frames: 8, width: 640, height: 352 },
					{ num_frames: 3, width: 1152, height: 640 }
				],
				2
			)
		).toBe(2 * 2 * 1152 * 640);
	});

	it('seeds settings inside the model ranges', () => {
		expect(seedVideoInputSettings(info)).toEqual({
			mode: 'frames',
			maxFrames: 32,
			fps: 2,
			tokensPerFrame: 768
		});
		expect(
			resolveVideoInputSettings(info, { maxFrames: 5000, fps: 100, tokensPerFrame: 7 })
		).toEqual({
			mode: 'frames',
			maxFrames: 768,
			fps: 8,
			tokensPerFrame: 64
		});
	});

	it('keeps settings stored before the Sampling FPS slider existed (fps → model default)', () => {
		const resolved = resolveVideoInputSettings(info, { maxFrames: 16, tokensPerFrame: 256 });
		expect(resolved).toEqual({ mode: 'frames', maxFrames: 16, fps: 2, tokensPerFrame: 256 });
		expect(resolveVideoInputSettings(info, { fps: 0.3 }).fps).toBe(0.25);
	});
});
