// GPU kernel: upload u8 RGB image, cast to f32 and scale by 1/255 in the
// shader.  This replaces the CPU cast loop and eliminates the intermediate
// Vec<f32> allocation entirely.
//
// The u8 bytes are packed 4-per-u32 in the storage buffer (WGSL has no
// native u8 array type).  Each thread unpacks one u32, extracts the bytes
// it owns, converts to f32, and writes to the output f32 buffer.

use crate::allocator::WgpuAllocator;
use crate::error::WgpuError;
use crate::ops::create_storage_buffer;
use crate::session::WgpuSession;
use crate::shader::{PipelineKey, WgslShader, CAST_U8_TO_F32};
use crate::transfer::wrap_gpu_buffer;
use kornia_image::Image;

// ── Immediates ────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CastParams {
    numel: u32, // total number of f32 elements (width * height * channels)
    scale: f32, // 1.0 / 255.0
    _pad: [u32; 2],
}

// input_u8 is array<u32> — four u8 bytes packed little-endian per u32.
// output_f32 is array<f32> — one f32 per original u8 byte.
//
// Thread i handles output element i:
//   word  = i / 4   → which u32 in the input
//   shift = (i % 4) * 8  → which byte within that u32
//   byte  = (word >> shift) & 0xFF
//   out[i] = f32(byte) * scale

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

// ── Dispatch ──────────────────────────────────────────────────────────────────

/// Upload a u8 image directly to the GPU and cast to f32 in a compute shader.
///
/// Replaces the CPU pattern:
///   `frame.as_slice().iter().map(|&v| v as f32 / 255.0).collect()`
///
/// The u8 bytes never materialise as a `Vec<f32>` on the CPU — they are
/// written to a `array<u32>` staging buffer and the cast happens entirely
/// on the GPU.
pub fn cast_u8_to_f32_gpu<const C: usize>(
    session: &WgpuSession,
    cpu_image: &Image<u8, C, impl kornia_image::allocator::ImageAllocator>,
) -> Result<Image<f32, C, WgpuAllocator>, WgpuError> {
    let size = cpu_image.size();
    let numel = size.width * size.height * C;

    let device_arc = session.raw_device_arc();
    let device = &device_arc.device;
    let queue = &device_arc.queue;

    // ── Upload u8 bytes packed into a u32 storage buffer ─────────────────────
    //
    // WGSL storage buffers require u32-aligned size.
    let u32_count = numel.div_ceil(4);
    let upload_byte_size = (u32_count * 4) as wgpu::BufferAddress;

    let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("CastU8 input (u8 packed as u32)"),
        size: upload_byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // bytemuck::cast_slice requires the slice to be aligned to u32 size.
    // We pad to a multiple of 4 bytes to satisfy this.
    let raw = cpu_image.as_slice();
    if numel % 4 == 0 {
        queue.write_buffer(&input_buffer, 0, raw);
    } else {
        let mut padded = raw.to_vec();
        padded.resize(u32_count * 4, 0u8);
        queue.write_buffer(&input_buffer, 0, &padded);
    }

    // ── Output f32 buffer ─────────────────────────────────────────────────────
    let out_byte_size = (numel * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
    let output_buffer = create_storage_buffer(device, out_byte_size);

    // ── Pipeline ──────────────────────────────────────────────────────────────
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

    // ── Dispatch ──────────────────────────────────────────────────────────────
    let params = CastParams {
        numel: numel as u32,
        scale: 1.0 / 255.0,
        _pad: [0, 0],
    };

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("CastU8ToF32 BindGroup"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("CastU8ToF32 encoder"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.set_immediates(0, bytemuck::bytes_of(&params));
        let wg_x = (numel as u32).div_ceil(64);
        cpass.dispatch_workgroups(wg_x, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    wrap_gpu_buffer(size, output_buffer, device_arc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transfer::image_to_cpu;
    use kornia_image::{allocator::CpuAllocator, Image, ImageSize};

    #[test]
    fn test_cast_u8_to_f32_values() {
        let session = pollster::block_on(WgpuSession::new()).unwrap();

        let size = ImageSize {
            width: 2,
            height: 2,
        };
        let data: Vec<u8> = vec![0, 128, 255, 64, 32, 16, 200, 100, 50, 255, 0, 127];
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
        let session = pollster::block_on(WgpuSession::new()).unwrap();

        // 720p RGB — real video frame size
        let size = ImageSize {
            width: 1280,
            height: 720,
        };
        let data: Vec<u8> = (0..1280 * 720 * 3).map(|i| (i % 256) as u8).collect();
        let cpu_image = Image::<u8, 3, _>::new(size, data, CpuAllocator).unwrap();

        let gpu_f32 = cast_u8_to_f32_gpu(&session, &cpu_image).unwrap();
        let _ = session
            .raw_device()
            .poll(wgpu::PollType::wait_indefinitely());
        let cpu_f32 = image_to_cpu(&session, &gpu_f32).unwrap();

        // All values must be in [0, 1]
        assert!(cpu_f32.as_slice().iter().all(|&v| (0.0..=1.0).contains(&v)));
    }
}
