import { describe, expect, it } from 'vitest';
import {
	maxPixelsFor,
	planVideo,
	resolveVideoInputSettings,
	sampleTimes,
	seedVideoInputSettings,
	smartResize,
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
		expect(plan.tokens).toBe(912);
		expect(plan.fps).toBeCloseTo(2.0);
	});

	it('scales the frame count with the Sampling FPS setting', () => {
		const plan = (fps: number, maxFrames = 32) =>
			planVideo(
				{ duration: 4, width: 640, height: 352 },
				info,
				resolveVideoInputSettings(info, { maxFrames, fps, tokensPerFrame: 768 })
			);

		expect(plan(4).n).toBe(16);
		expect(plan(4).fps).toBeCloseTo(4.0);

		// floor(4 s × 0.5) = 2 → min_frames 4 → still 4 after temporal-patch rounding
		expect(plan(0.5).n).toBe(4);
		expect(plan(0.5).fps).toBeCloseTo(1.0);

		// Max Frames still caps: 4 s × 8 fps = 32 → 8
		expect(plan(8, 8).n).toBe(8);
		expect(plan(8, 8).fps).toBeCloseTo(2.0);
	});

	it('clamps to the max-frames slider and keeps n a multiple of the temporal patch', () => {
		const plan = planVideo(
			{ duration: 60, width: 640, height: 352 },
			info,
			resolveVideoInputSettings(info, { maxFrames: 30, tokensPerFrame: 768 })
		);
		expect(plan.n).toBe(30);
		expect(plan.fps).toBeCloseTo(0.5);
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

describe('helpers', () => {
	it('samples frame times at the centre of each slot', () => {
		expect(sampleTimes(4, 8)).toEqual([0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75]);
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
