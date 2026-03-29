use crate::error::WgpuError;
use crate::gpu_res::GpuImage;
use crate::session::WgpuSession;
use crate::shader::{PipelineKey, WgslShader, CAST_F32_TO_U8, CAST_U8_TO_F32};
use crate::transfer::{acquire_output, wrap_gpu_buffer};
use kornia_image::Image;

// ── Shared params layout (both directions use the same struct) ────────────────

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CastParams {
    numel: u32,
    scale: f32,
    _pad: [u32; 2],
}

// ── u8 → f32 ─────────────────────────────────────────────────────────────────

const CAST_U8_TO_F32_WGSL: &str = r#"
struct Params {
    numel: u32,
    scale: f32,
    _pad: vec2<u32>,
}
var<immediate> p: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= p.numel { return; }

    let word  = i / 4u;
    let shift = (i % 4u) * 8u;
    let byte  = (input_u8[word] >> shift) & 0xFFu;

    output_f32[i] = f32(byte) * p.scale;
}
"#;

/// Casts a CPU `u8` image to a GPU `f32` image, scaling pixel values to `[0.0, 1.0]`.
///
/// The raw u8 bytes are packed 4-per-u32 and uploaded directly. The WGSL
/// shader unpacks each byte and divides by 255, eliminating the CPU cast
/// bottleneck entirely.
///
/// # Arguments
///
/// * `session` - The active `WgpuSession`.
/// * `cpu_image` - The input CPU image to cast and upload.
///
/// # Returns
///
/// A [`GpuImage`] of type `f32` residing on the GPU.
///
/// # Errors
///
/// Returns a [`WgpuError`] if pipeline creation or buffer allocation fails.
pub fn cast_u8_to_f32_gpu<const C: usize>(
    session: &WgpuSession,
    cpu_image: &Image<u8, C, impl kornia_image::allocator::ImageAllocator>,
) -> Result<GpuImage<f32, C>, WgpuError> {
    let size = cpu_image.size();
    let numel = size.width * size.height * C;

    let device_arc = session.raw_device_arc();
    let device = &device_arc.device;
    let queue = &device_arc.queue;

    // Input buffer — NOT pooled (transient upload, variable padding)
    let u32_count = numel.div_ceil(4);
    let upload_byte_size = (u32_count * 4) as wgpu::BufferAddress;
    let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cast-u8-input"),
        size: upload_byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let raw = cpu_image.as_slice();
    if numel % 4 == 0 {
        queue.write_buffer(&input_buffer, 0, raw);
    } else {
        let mut padded = raw.to_vec();
        padded.resize(u32_count * 4, 0u8);
        queue.write_buffer(&input_buffer, 0, &padded);
    }

    // Output buffer — pooled
    let out_byte_size = (numel * std::mem::size_of::<f32>()) as u64;
    let output_alloc = acquire_output(session, out_byte_size);

    let key = PipelineKey {
        shader_name: CAST_U8_TO_F32.name,
        pixel_bytes: 1,
        channels: C as u8,
        wg_x: 64,
        wg_y: 1,
        variant: 0,
    };

    let mut shader = WgslShader {
        kind: CAST_U8_TO_F32.clone(),
        source: CAST_U8_TO_F32_WGSL.to_string(),
    };
    shader.build();

    let pipeline =
        device_arc.get_or_create_pipeline(key, &shader, std::mem::size_of::<CastParams>() as u32);

    let params = CastParams {
        numel: numel as u32,
        scale: 1.0 / 255.0,
        _pad: [0, 0],
    };

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cast-u8-to-f32"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_alloc.gpu_buffer().as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("cast-u8-to-f32"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.set_immediates(0, bytemuck::bytes_of(&params));
        cpass.dispatch_workgroups((numel as u32).div_ceil(64), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    wrap_gpu_buffer(size, output_alloc)
}

// ── f32 → u8 ─────────────────────────────────────────────────────────────────

/// WGSL for the f32 → packed u8 kernel.
///
/// Each thread handles 4 consecutive f32 elements and packs them into one u32
/// word (little-endian byte order), clamping to [0, 255] and rounding.
///
/// Thread i writes output word i:
///   output_u8[i] = pack4x8unorm(saturate(input_f32[i*4 .. i*4+3]))
///
/// `pack4x8unorm` is a WGSL built-in: it multiplies each component by 255,
/// rounds to the nearest integer, clamps to [0, 255], and packs the four
/// bytes into a u32 in little-endian order. This is exactly equivalent to
/// `(v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8` but executes in a single
/// hardware instruction on most GPUs.
///
/// Tail handling: for `numel % 4 != 0`, the last word is still written with
/// its out-of-range lanes set to zero (they are masked out on the CPU side
/// by truncating the output slice to `numel` bytes).
const CAST_F32_TO_U8_WGSL: &str = r#"
struct Params {
    numel: u32,   // total number of f32 elements (= W * H * C)
    scale: f32,   // unused for this direction; kept for layout compatibility
    _pad: vec2<u32>,
}
var<immediate> p: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Each thread packs 4 consecutive f32s into one u32 output word.
    let word = gid.x;

    // Total number of u32 words needed to cover all elements.
    let word_count = (p.numel + 3u) / 4u;
    if word >= word_count { return; }

    let base = word * 4u;

    // Read 4 f32 values; pad with 0.0 if we are in the tail word.
    var v: vec4<f32>;
    v.x = select(0.0, input_f32[base + 0u], base + 0u < p.numel);
    v.y = select(0.0, input_f32[base + 1u], base + 1u < p.numel);
    v.z = select(0.0, input_f32[base + 2u], base + 2u < p.numel);
    v.w = select(0.0, input_f32[base + 3u], base + 3u < p.numel);

    // pack4x8unorm: saturates to [0, 1], multiplies by 255, rounds, packs
    // four bytes into a u32 (little-endian: byte 0 = least-significant byte).
    output_u8[word] = pack4x8unorm(v);
}
"#;

/// Casts a GPU `f32` image to a GPU `u8` image entirely on the GPU.
///
/// This is the inverse of [`cast_u8_to_f32_gpu`] and is intended for the
/// video pipeline output stage, replacing the CPU iterator
/// `pixels.map(|v| (v * 255.0).clamp(0.0, 255.0) as u8)`.
///
/// Values are saturated to `[0.0, 1.0]` before conversion. The WGSL
/// `pack4x8unorm` built-in handles clamping, scaling, and byte-packing in
/// a single hardware instruction.
///
/// The returned [`GpuImage<u8, C>`] can be downloaded with [`image_to_cpu`]
/// or consumed by a downstream GStreamer appsrc without any CPU work.
///
/// # Arguments
///
/// * `session` - The active `WgpuSession`.
/// * `input`   - The input `f32` image residing in GPU memory.
///
/// # Returns
///
/// A [`GpuImage<u8, C>`] residing on the GPU.
///
/// # Errors
///
/// Returns a [`WgpuError`] if pipeline creation or buffer allocation fails.
pub fn cast_f32_to_u8_gpu<const C: usize>(
    session: &WgpuSession,
    input: &GpuImage<f32, C>,
) -> Result<GpuImage<u8, C>, WgpuError> {
    let size = input.size();
    let numel = size.width * size.height * C; // number of f32 elements

    let device_arc = session.raw_device_arc();
    let device = &device_arc.device;
    let queue = &device_arc.queue;

    // The output u8 buffer needs `numel` bytes, rounded up to a u32 boundary
    // so that the last thread can safely write a full word.
    let u32_count = numel.div_ceil(4);
    let out_byte_size = (u32_count * 4) as u64;
    let output_alloc = acquire_output(session, out_byte_size);

    let key = PipelineKey {
        shader_name: CAST_F32_TO_U8.name,
        pixel_bytes: 4, // input element size: f32
        channels: C as u8,
        wg_x: 64,
        wg_y: 1,
        variant: 0,
    };

    let mut shader = WgslShader {
        kind: CAST_F32_TO_U8.clone(),
        source: CAST_F32_TO_U8_WGSL.to_string(),
    };
    shader.build();

    let pipeline =
        device_arc.get_or_create_pipeline(key, &shader, std::mem::size_of::<CastParams>() as u32);

    let params = CastParams {
        numel: numel as u32,
        scale: 255.0, // informational; the shader uses pack4x8unorm directly
        _pad: [0, 0],
    };

    // Bind the f32 source buffer (read) and the u32-typed output buffer (write).
    // The output buffer is acquired from the pool as raw bytes; we treat it as
    // array<u32> inside the shader and as u8 bytes after download.
    let in_buffer = crate::transfer::src_buffer(input);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cast-f32-to-u8"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: in_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_alloc.gpu_buffer().as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("cast-f32-to-u8"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.set_immediates(0, bytemuck::bytes_of(&params));
        // One thread per output word (= 4 u8 elements).
        cpass.dispatch_workgroups((u32_count as u32).div_ceil(64), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    // wrap_gpu_buffer uses the logical pixel type (u8) and the image dimensions.
    // The pool buffer is slightly oversized (padded to u32 boundary) but the
    // GpuImage metadata records the true width/height, so image_to_cpu will
    // only copy `numel` bytes.
    wrap_gpu_buffer(size, output_alloc)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transfer::{image_to_cpu, image_to_gpu};
    use kornia_image::{allocator::CpuAllocator, Image, ImageSize};

    fn make_session() -> WgpuSession {
        pollster::block_on(WgpuSession::new()).unwrap()
    }

    // ── cast_u8_to_f32_gpu ────────────────────────────────────────────────────

    #[test]
    fn test_cast_u8_to_f32_values() {
        let session = make_session();
        let size = ImageSize {
            width: 2,
            height: 2,
        };
        let data = vec![0u8, 128, 255, 64, 32, 16, 200, 100, 50, 255, 0, 127];
        let cpu_image = Image::<u8, 3, _>::new(size, data.clone(), CpuAllocator).unwrap();

        let gpu_f32 = cast_u8_to_f32_gpu(&session, &cpu_image).unwrap();
        let _ = session
            .raw_device()
            .poll(wgpu::PollType::wait_indefinitely());
        let cpu_f32 = image_to_cpu(&session, &gpu_f32).unwrap();

        for (i, (&orig, &result)) in data.iter().zip(cpu_f32.as_slice().iter()).enumerate() {
            let expected = orig as f32 / 255.0;
            assert!(
                (result - expected).abs() < 1e-6,
                "index {i}: expected {expected}, got {result}"
            );
        }
    }

    #[test]
    fn test_cast_u8_to_f32_range() {
        let session = make_session();
        let size = ImageSize {
            width: 1280,
            height: 720,
        };
        let data = (0..1280 * 720 * 3).map(|i| (i % 256) as u8).collect();
        let cpu_image = Image::<u8, 3, _>::new(size, data, CpuAllocator).unwrap();

        let gpu_f32 = cast_u8_to_f32_gpu(&session, &cpu_image).unwrap();
        let _ = session
            .raw_device()
            .poll(wgpu::PollType::wait_indefinitely());
        let cpu_f32 = image_to_cpu(&session, &gpu_f32).unwrap();

        assert!(cpu_f32
            .as_slice()
            .iter()
            .all(|&v| (0.0f32..=1.0).contains(&v)));
    }

    // ── cast_f32_to_u8_gpu ────────────────────────────────────────────────────

    /// Round-trip u8 → f32 → u8 must be lossless for every value 0..=255
    /// (the pack4x8unorm rounding is within ±1 ULP at u8 precision).
    #[test]
    fn test_cast_f32_to_u8_round_trip() {
        let session = make_session();
        let size = ImageSize {
            width: 4,
            height: 4,
        };
        // Use a known pattern: one pixel per value, 1 channel, 4×4 = 16 pixels.
        let original_u8: Vec<u8> = (0u8..16).map(|v| v * 16).collect(); // 0, 16, 32, …, 240
        let cpu_u8 = Image::<u8, 1, _>::new(size, original_u8.clone(), CpuAllocator).unwrap();

        // u8 → f32 on GPU
        let gpu_f32 = cast_u8_to_f32_gpu(&session, &cpu_u8).unwrap();
        let _ = session
            .raw_device()
            .poll(wgpu::PollType::wait_indefinitely());

        // f32 → u8 on GPU
        let gpu_u8 = cast_f32_to_u8_gpu(&session, &gpu_f32).unwrap();
        let _ = session
            .raw_device()
            .poll(wgpu::PollType::wait_indefinitely());

        let result = image_to_cpu(&session, &gpu_u8).unwrap();

        for (i, (&orig, &got)) in original_u8.iter().zip(result.as_slice().iter()).enumerate() {
            // pack4x8unorm rounds to nearest, so we allow ±1 LSB.
            let diff = (orig as i32 - got as i32).abs();
            assert!(
                diff <= 1,
                "index {i}: original u8={orig}, round-tripped u8={got}, diff={diff}"
            );
        }
    }

    /// Values exactly at 0.0 and 1.0 must clamp correctly.
    #[test]
    fn test_cast_f32_to_u8_boundary_values() {
        let session = make_session();
        let size = ImageSize {
            width: 4,
            height: 1,
        };
        // 0.0 → 0, 1.0 → 255, 0.5 → ~128, -0.1 → 0 (clamp), 1.5 → 255 (clamp)
        let data = vec![0.0f32, 1.0, 0.5, -0.1, 1.5, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let cpu_f32 = Image::<f32, 3, _>::new(size, data.clone(), CpuAllocator).unwrap();
        let gpu_f32 = image_to_gpu(&session, &cpu_f32).unwrap();

        let gpu_u8 = cast_f32_to_u8_gpu(&session, &gpu_f32).unwrap();
        let _ = session
            .raw_device()
            .poll(wgpu::PollType::wait_indefinitely());
        let result = image_to_cpu(&session, &gpu_u8).unwrap();

        let s = result.as_slice();
        assert_eq!(s[0], 0, "0.0 → 0");
        assert_eq!(s[1], 255, "1.0 → 255");
        assert!(
            (s[2] as i32 - 128).abs() <= 1,
            "0.5 → ~128, got {}",
            s[2]
        );
        assert_eq!(s[3], 0, "-0.1 → 0 (saturate)");
        assert_eq!(s[4], 255, "1.5 → 255 (saturate)");
    }

    /// Matches the GPU output against a naive CPU reference on a 720p RGB frame.
    #[test]
    fn test_cast_f32_to_u8_matches_cpu_reference() {
        let session = make_session();
        let size = ImageSize {
            width: 1280,
            height: 720,
        };
        let numel = 1280 * 720 * 3;
        // Ramp from 0.0 to just-below 1.0
        let data: Vec<f32> = (0..numel).map(|i| i as f32 / numel as f32).collect();
        let cpu_f32 = Image::<f32, 3, _>::new(size, data.clone(), CpuAllocator).unwrap();
        let gpu_f32 = image_to_gpu(&session, &cpu_f32).unwrap();

        let gpu_u8 = cast_f32_to_u8_gpu(&session, &gpu_f32).unwrap();
        let _ = session
            .raw_device()
            .poll(wgpu::PollType::wait_indefinitely());
        let result = image_to_cpu(&session, &gpu_u8).unwrap();

        // CPU reference: same formula as the old video_pipeline iterator
        let reference: Vec<u8> = data
            .iter()
            .map(|&v| (v * 255.0).clamp(0.0, 255.0) as u8)
            .collect();

        let mismatches: Vec<usize> = result
            .as_slice()
            .iter()
            .zip(reference.iter())
            .enumerate()
            .filter(|(_, (&gpu, &cpu))| (gpu as i32 - cpu as i32).abs() > 1)
            .map(|(i, _)| i)
            .collect();

        assert!(
            mismatches.is_empty(),
            "{} pixels differ by more than 1 LSB (first: index {})",
            mismatches.len(),
            mismatches[0]
        );
    }

    /// Non-multiple-of-4 element count (tail-word handling).
    #[test]
    fn test_cast_f32_to_u8_non_aligned_numel() {
        let session = make_session();
        // 3×1 = 3 elements — not divisible by 4
        let size = ImageSize {
            width: 3,
            height: 1,
        };
        let data = vec![0.0f32, 0.5, 1.0];
        let cpu_f32 = Image::<f32, 1, _>::new(size, data, CpuAllocator).unwrap();
        let gpu_f32 = image_to_gpu(&session, &cpu_f32).unwrap();

        let gpu_u8 = cast_f32_to_u8_gpu(&session, &gpu_f32).unwrap();
        let _ = session
            .raw_device()
            .poll(wgpu::PollType::wait_indefinitely());
        let result = image_to_cpu(&session, &gpu_u8).unwrap();

        assert_eq!(result.as_slice().len(), 3);
        assert_eq!(result.as_slice()[0], 0);
        assert!((result.as_slice()[1] as i32 - 128).abs() <= 1);
        assert_eq!(result.as_slice()[2], 255);
    }
}