// src/ops/image/resize.rs
use crate::allocator::WgpuAllocator;
use crate::error::WgpuError;
use crate::ops::{compute_1in_1out, create_storage_buffer};
use crate::session::WgpuSession;
use crate::shader::ShaderKind;
use crate::transfer::{src_buffer, wrap_gpu_buffer};
use kornia_image::{Image, ImageSize};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ResizeImmediates {
    pub in_width: u32,
    pub in_height: u32,
    pub out_width: u32,
    pub out_height: u32,
    pub channels: u32,
    pub _pad: u32,
}

const RESIZE_WGSL: &str = r#"
struct Immediates {
    in_width: u32,
    in_height: u32,
    out_width: u32,
    out_height: u32,
    channels: u32,
    _pad: u32,
}
var<immediate> params: Immediates;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.out_width || y >= params.out_height) { return; }

    let scale_x = f32(params.in_width)  / f32(params.out_width);
    let scale_y = f32(params.in_height) / f32(params.out_height);

    let src_x = min(u32(f32(x) * scale_x), params.in_width  - 1u);
    let src_y = min(u32(f32(y) * scale_y), params.in_height - 1u);

    let src_base = (src_y * params.in_width  + src_x) * params.channels;
    let dst_base = (y    * params.out_width  + x)     * params.channels;

    for (var c = 0u; c < params.channels; c++) {
        output_buf[dst_base + c] = input_buf[src_base + c];
    }
}
"#;

const RESIZE_BILINEAR_WGSL: &str = r#"
struct PushConstants {
    in_width: u32,
    in_height: u32,
    out_width: u32,
    out_height: u32,
    channels: u32,
    _pad: u32,
}
var<immediate> params: PushConstants;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.out_width || y >= params.out_height) { return; }

    let scale_x = f32(params.in_width)  / f32(params.out_width);
    let scale_y = f32(params.in_height) / f32(params.out_height);

    let src_x = (f32(x) + 0.5) * scale_x - 0.5;
    let src_y = (f32(y) + 0.5) * scale_y - 0.5;

    let x1 = u32(max(0.0, floor(src_x)));
    let y1 = u32(max(0.0, floor(src_y)));
    let x2 = min(x1 + 1u, params.in_width  - 1u);
    let y2 = min(y1 + 1u, params.in_height - 1u);

    let wx = max(0.0, src_x - f32(x1));
    let wy = max(0.0, src_y - f32(y1));

    let dst_base = (y * params.out_width + x) * params.channels;

    for (var c = 0u; c < params.channels; c++) {
        let p11 = input_buf[(y1 * params.in_width + x1) * params.channels + c];
        let p12 = input_buf[(y1 * params.in_width + x2) * params.channels + c];
        let p21 = input_buf[(y2 * params.in_width + x1) * params.channels + c];
        let p22 = input_buf[(y2 * params.in_width + x2) * params.channels + c];

        output_buf[dst_base + c] = mix(mix(p11, p12, wx), mix(p21, p22, wx), wy);
    }
}
"#;

pub fn resize_nearest_f32<const C: usize>(
    session: &WgpuSession,
    input: &Image<f32, C, WgpuAllocator>,
    new_size: ImageSize,
) -> Result<Image<f32, C, WgpuAllocator>, WgpuError> {
    let numel = new_size.width * new_size.height * C;
    let byte_size = (numel * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
    let out_buffer = create_storage_buffer(session.raw_device(), byte_size);

    let immediates = ResizeImmediates {
        in_width: input.size().width as u32,
        in_height: input.size().height as u32,
        out_width: new_size.width as u32,
        out_height: new_size.height as u32,
        channels: C as u32,
        _pad: 0,
    };

    compute_1in_1out(
        session,
        ShaderKind::ResizeNearest,
        RESIZE_WGSL,
        src_buffer(input),
        &out_buffer,
        &immediates,
        (new_size.width as u32, new_size.height as u32),
        (16, 16),
        4,
        C as u8,
    );

    wrap_gpu_buffer(new_size, out_buffer, session.raw_device_arc())
}

pub fn resize_bilinear_f32<const C: usize>(
    session: &WgpuSession,
    input: &Image<f32, C, WgpuAllocator>,
    new_size: ImageSize,
) -> Result<Image<f32, C, WgpuAllocator>, WgpuError> {
    let numel = new_size.width * new_size.height * C;
    let byte_size = (numel * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
    let out_buffer = create_storage_buffer(session.raw_device(), byte_size);

    let immediates = ResizeImmediates {
        in_width: input.size().width as u32,
        in_height: input.size().height as u32,
        out_width: new_size.width as u32,
        out_height: new_size.height as u32,
        channels: C as u32,
        _pad: 0,
    };

    compute_1in_1out(
        session,
        ShaderKind::ResizeBilinear,
        RESIZE_BILINEAR_WGSL,
        src_buffer(input),
        &out_buffer,
        &immediates,
        (new_size.width as u32, new_size.height as u32),
        (16, 16),
        4,
        C as u8,
    );

    wrap_gpu_buffer(new_size, out_buffer, session.raw_device_arc())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transfer::{image_to_cpu, image_to_gpu};
    use kornia_tensor::allocator::CpuAllocator;

    #[test]
    fn test_resize_nearest_f32_upscale() {
        let session = pollster::block_on(WgpuSession::new()).unwrap();
        let in_size = ImageSize {
            width: 2,
            height: 2,
        };
        let original =
            Image::<f32, 1, _>::new(in_size, vec![1.0, 2.0, 3.0, 4.0], CpuAllocator).unwrap();
        let gpu = image_to_gpu(&session, &original).unwrap();
        let resized_gpu = resize_nearest_f32(
            &session,
            &gpu,
            ImageSize {
                width: 4,
                height: 4,
            },
        )
        .unwrap();
        let resized_cpu = image_to_cpu(&session, &resized_gpu).unwrap();
        let expected: Vec<f32> = vec![
            1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
        ];
        assert_eq!(resized_cpu.as_slice(), expected.as_slice());
    }

    #[test]
    fn test_resize_bilinear_f32_upscale() {
        let session = pollster::block_on(WgpuSession::new()).unwrap();
        let in_size = ImageSize {
            width: 2,
            height: 2,
        };
        let original =
            Image::<f32, 1, _>::new(in_size, vec![1.0, 2.0, 3.0, 4.0], CpuAllocator).unwrap();
        let gpu = image_to_gpu(&session, &original).unwrap();
        let resized_gpu = resize_bilinear_f32(
            &session,
            &gpu,
            ImageSize {
                width: 4,
                height: 4,
            },
        )
        .unwrap();
        let resized_cpu = image_to_cpu(&session, &resized_gpu).unwrap();
        let expected: Vec<f32> = vec![
            1.00, 1.25, 1.75, 2.00, 1.50, 1.75, 2.25, 2.50, 2.50, 2.75, 3.25, 3.50, 3.00, 3.25,
            3.75, 4.00,
        ];
        for (i, (&res, &exp)) in resized_cpu
            .as_slice()
            .iter()
            .zip(expected.iter())
            .enumerate()
        {
            assert!(
                (res - exp).abs() < 1e-4,
                "Mismatch at {i}: expected {exp}, got {res}"
            );
        }
    }

    #[test]
    fn test_resize_bilinear_3channel() {
        // Verifies the channel loop in the shader works for RGB images
        let session = pollster::block_on(WgpuSession::new()).unwrap();
        let in_size = ImageSize {
            width: 2,
            height: 2,
        };
        // 2x2 RGB: each pixel is (r, g, b)
        // pixel(0,0)=(1,0,0), pixel(1,0)=(0,1,0), pixel(0,1)=(0,0,1), pixel(1,1)=(1,1,1)
        let data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let original = Image::<f32, 3, _>::new(in_size, data, CpuAllocator).unwrap();
        let gpu = image_to_gpu(&session, &original).unwrap();
        let resized_gpu = resize_bilinear_f32(
            &session,
            &gpu,
            ImageSize {
                width: 4,
                height: 4,
            },
        )
        .unwrap();
        let resized_cpu = image_to_cpu(&session, &resized_gpu).unwrap();
        // Output must have correct size and all values in [0,1]
        assert_eq!(resized_cpu.as_slice().len(), 4 * 4 * 3);
        assert!(resized_cpu
            .as_slice()
            .iter()
            .all(|&v| (0.0..=1.0).contains(&v)));
    }
}
