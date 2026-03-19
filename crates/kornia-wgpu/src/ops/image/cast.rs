use crate::error::WgpuError;
use crate::gpu_res::GpuImage;
use crate::session::WgpuSession;
use crate::shader::{PipelineKey, WgslShader, CAST_U8_TO_F32};
use crate::transfer::{acquire_output, wrap_gpu_buffer};
use kornia_image::Image;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CastParams {
    numel: u32,
    scale: f32,
    _pad: [u32; 2],
}

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

/// Casts a GPU image of type `u8` to `f32`, scaling pixel values to `[0.0, 1.0]`.
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

    // Input buffer — NOT pooled (transient upload)
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
        let session = pollster::block_on(WgpuSession::new()).unwrap();
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
}
