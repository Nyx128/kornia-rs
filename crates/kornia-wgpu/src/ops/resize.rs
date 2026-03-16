// src/ops/resize.rs
use super::{compute_1in_1out, create_storage_buffer};
use crate::allocator::WgpuAllocator;
use crate::error::WgpuError;
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
}

const RESIZE_WGSL: &str = r#"
struct Immediates {
    in_width: u32,
    in_height: u32,
    out_width: u32,
    out_height: u32,
}
var<immediate> params: Immediates;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    // Bounds check based on the target image size
    if (x >= params.out_width || y >= params.out_height) {
        return;
    }
    
    // Calculate nearest neighbor source coordinates
    let scale_x = f32(params.in_width) / f32(params.out_width);
    let scale_y = f32(params.in_height) / f32(params.out_height);
    
    let src_x = min(u32(f32(x) * scale_x), params.in_width - 1u);
    let src_y = min(u32(f32(y) * scale_y), params.in_height - 1u);
    
    // 1-channel flat index calculation
    let src_idx = src_y * params.in_width + src_x;
    let dst_idx = y * params.out_width + x;
    
    output_buf[dst_idx] = input_buf[src_idx];
}
"#;

const RESIZE_BILINEAR_WGSL: &str = r#"
struct PushConstants {
    in_width: u32,
    in_height: u32,
    out_width: u32,
    out_height: u32,
}
var<immediate> params: PushConstants;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.out_width || y >= params.out_height) { return; }
    
    let scale_x = f32(params.in_width) / f32(params.out_width);
    let scale_y = f32(params.in_height) / f32(params.out_height);
    
    // Calculate source coordinates (center-aligned)
    let src_x = (f32(x) + 0.5) * scale_x - 0.5;
    let src_y = (f32(y) + 0.5) * scale_y - 0.5;
    
    // Get the 4 neighboring pixel coordinates
    let x1 = u32(max(0.0, floor(src_x)));
    let y1 = u32(max(0.0, floor(src_y)));
    let x2 = min(x1 + 1u, params.in_width - 1u);
    let y2 = min(y1 + 1u, params.in_height - 1u);
    
    // Calculate fractional weights
    let wx = max(0.0, src_x - f32(x1));
    let wy = max(0.0, src_y - f32(y1));
    
    // Read the 4 pixels
    let p11 = input_buf[y1 * params.in_width + x1];
    let p12 = input_buf[y1 * params.in_width + x2];
    let p21 = input_buf[y2 * params.in_width + x1];
    let p22 = input_buf[y2 * params.in_width + x2];
    
    // Interpolate
    let top = mix(p11, p12, wx);
    let bottom = mix(p21, p22, wx);
    let final_val = mix(top, bottom, wy);
    
    let dst_idx = y * params.out_width + x;
    output_buf[dst_idx] = final_val;
}
"#;

pub fn resize_nearest_f32(
    session: &WgpuSession,
    input: &Image<f32, 1, WgpuAllocator>,
    new_size: ImageSize,
) -> Result<Image<f32, 1, WgpuAllocator>, WgpuError> {
    let numel = new_size.width * new_size.height * 1;
    let byte_size = (numel * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
    let out_buffer = create_storage_buffer(session.raw_device(), byte_size);

    // Setup Immediate Data
    let immediates = ResizeImmediates {
        in_width: input.size().width as u32,
        in_height: input.size().height as u32,
        out_width: new_size.width as u32,
        out_height: new_size.height as u32,
    };

    // Dispatch the compute pass using our generic helper!
    compute_1in_1out(
        session,
        ShaderKind::ResizeNearest,
        RESIZE_WGSL,
        src_buffer(input),
        &out_buffer,
        &immediates,
        (new_size.width as u32, new_size.height as u32), // dispatch size
        (16, 16),                                        // workgroup size
        4,                                               // f32 is 4 bytes
        1,                                               // 1 channel
    );

    //Wrap the output buffer back into an Image
    wrap_gpu_buffer(new_size, out_buffer, session.raw_device_arc())
}

pub fn resize_bilinear_f32(
    session: &WgpuSession,
    input: &Image<f32, 1, WgpuAllocator>,
    new_size: ImageSize,
) -> Result<Image<f32, 1, WgpuAllocator>, WgpuError> {
    let numel = new_size.width * new_size.height * 1;
    let byte_size = (numel * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
    let out_buffer = super::create_storage_buffer(session.raw_device(), byte_size);

    let immediates = ResizeImmediates {
        in_width: input.size().width as u32,
        in_height: input.size().height as u32,
        out_width: new_size.width as u32,
        out_height: new_size.height as u32,
    };

    super::compute_1in_1out(
        session,
        ShaderKind::ResizeBilinear,
        RESIZE_BILINEAR_WGSL,
        crate::transfer::src_buffer(input),
        &out_buffer,
        &immediates,
        (new_size.width as u32, new_size.height as u32),
        (16, 16),
        4,
        1,
    );

    crate::transfer::wrap_gpu_buffer(new_size, out_buffer, session.raw_device_arc())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transfer::{image_to_cpu, image_to_gpu};
    use kornia_tensor::allocator::CpuAllocator;

    #[test]
    fn test_resize_nearest_f32_upscale() {
        let session = pollster::block_on(WgpuSession::new()).unwrap();

        //Create a 2x2 CPU image
        let in_size = ImageSize {
            width: 2,
            height: 2,
        };
        let cpu_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let original_image = Image::<f32, 1, _>::new(in_size, cpu_data, CpuAllocator).unwrap();

        // Upload to GPU
        let gpu_image = image_to_gpu(&session, &original_image).unwrap();

        // Resize to 4x4 on the GPU
        let out_size = ImageSize {
            width: 4,
            height: 4,
        };
        let resized_gpu = resize_nearest_f32(&session, &gpu_image, out_size).unwrap();

        // Download the result back to CPU
        let resized_cpu = image_to_cpu(&session, &resized_gpu).unwrap();

        // Verify nearest neighbor logic (each 1x1 pixel becomes a 2x2 block)
        let expected_data: Vec<f32> = vec![
            1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
        ];

        assert_eq!(resized_cpu.as_slice(), expected_data.as_slice());
    }

    #[test]
    fn test_resize_bilinear_f32_upscale() {
        let session = pollster::block_on(WgpuSession::new()).unwrap();

        // 1. Create a 2x2 CPU image
        let in_size = ImageSize {
            width: 2,
            height: 2,
        };
        let cpu_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let original_image = Image::<f32, 1, _>::new(in_size, cpu_data, CpuAllocator).unwrap();

        // 2. Upload to GPU
        let gpu_image = image_to_gpu(&session, &original_image).unwrap();

        // 3. Resize to 4x4 using our new Bilinear function
        let out_size = ImageSize {
            width: 4,
            height: 4,
        };
        let resized_gpu = resize_bilinear_f32(&session, &gpu_image, out_size).unwrap();

        // 4. Download the result back to CPU
        let resized_cpu = image_to_cpu(&session, &resized_gpu).unwrap();

        // 5. Verify the bilinear math!
        // Top row blends 1.0 to 2.0. Bottom row blends 3.0 to 4.0.
        // Columns blend the top row into the bottom row.
        let expected_data: Vec<f32> = vec![
            1.00, 1.25, 1.75, 2.00, 1.50, 1.75, 2.25, 2.50, 2.50, 2.75, 3.25, 3.50, 3.00, 3.25,
            3.75, 4.00,
        ];

        let result_data = resized_cpu.as_slice();

        // Use a small epsilon for floating-point comparisons just to be safe with GPU math
        for (i, (&res, &exp)) in result_data.iter().zip(expected_data.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-4,
                "Mismatch at index {}: expected {}, got {}",
                i,
                exp,
                res
            );
        }
    }
}
