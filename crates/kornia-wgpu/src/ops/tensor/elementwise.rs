use crate::allocator::WgpuAllocator;
use crate::error::WgpuError;
use crate::session::WgpuSession;
use crate::shader::{PipelineKey, ShaderKind, WgslShader};
use kornia_tensor::Tensor;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ElemParams {
    pub numel: u32,
    pub op_kind: u32, // 0=add, 1=sub, 2=mul, 3=div, ... 9=relu
    pub _pad: [u32; 2],
}

const ELEMENTWISE_WGSL: &str = r#"
struct Params {
    numel: u32,
    op_kind: u32,
    _pad: vec2<u32>,
}
var<immediate> p: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= p.numel { return; }

    let av = a[i];
    let bv = b[i];
    
    var res: f32 = av;
    
    switch p.op_kind {
        case 0u: { res = av + bv; }
        case 1u: { res = av - bv; }
        case 2u: { res = av * bv; }
        case 3u: { res = av / bv; }
        case 4u: { res = abs(av); }
        case 5u: { res = -av; }
        case 6u: { res = sqrt(av); }
        case 7u: { res = exp(av); }
        case 8u: { res = log(av); }
        case 9u: { res = max(0.0, av); } // ReLU
        default: { res = av; }
    }
    
    out[i] = res;
}
"#;

fn compute_tensor_elementwise(
    session: &WgpuSession,
    in_a: &wgpu::Buffer,
    in_b: &wgpu::Buffer,
    out_buffer: &wgpu::Buffer,
    params: &ElemParams,
) {
    let device_arc = session.raw_device_arc();
    let device = &device_arc.device;
    let queue = &device_arc.queue;

    let key = PipelineKey {
        shader_id: ShaderKind::TensorElementwise,
        pixel_bytes: 4,
        channels: 1,
        wg_x: 64,
        wg_y: 1,
        variant: 0,
    };

    let mut shader = WgslShader {
        kind: ShaderKind::TensorElementwise,
        source: ELEMENTWISE_WGSL.to_string(),
    };
    shader.build(); // now injects a, b, out declarations from bindings()

    let pipeline = device_arc.get_or_create_pipeline(
        key,
        &shader,
        std::mem::size_of::<ElemParams>() as u32,
        // no bgl_entries argument
    );

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Elementwise Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: in_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: in_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.set_immediates(0, bytemuck::bytes_of(params));

        let wg_x = params.numel.div_ceil(64);
        cpass.dispatch_workgroups(wg_x, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
}

pub fn add<const N: usize>(
    session: &WgpuSession,
    a: &Tensor<f32, N, WgpuAllocator>,
    b: &Tensor<f32, N, WgpuAllocator>,
) -> Result<Tensor<f32, N, WgpuAllocator>, WgpuError> {
    let numel = a.shape.iter().product::<usize>() as u32;
    let byte_size = (numel as usize * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
    let out_buffer = crate::ops::create_storage_buffer(session.raw_device(), byte_size);

    let params = ElemParams {
        numel,
        op_kind: 0,
        _pad: [0, 0],
    }; // 0 = add

    // Extract raw wgpu::Buffer from TensorStorage (assuming you have a helper for this like in images)
    let buf_a = crate::transfer::src_buffer_tensor(a);
    let buf_b = crate::transfer::src_buffer_tensor(b);

    compute_tensor_elementwise(session, buf_a, buf_b, &out_buffer, &params);

    crate::transfer::wrap_gpu_tensor(a.shape, a.strides, out_buffer, session.raw_device_arc())
}

/// Applies the Rectified Linear Unit function (Unary)
pub fn relu<const N: usize>(
    session: &WgpuSession,
    a: &Tensor<f32, N, WgpuAllocator>,
) -> Result<Tensor<f32, N, WgpuAllocator>, WgpuError> {
    let numel = a.shape.iter().product::<usize>() as u32;
    let byte_size = (numel as usize * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
    let out_buffer = crate::ops::create_storage_buffer(session.raw_device(), byte_size);

    let params = ElemParams {
        numel,
        op_kind: 9,
        _pad: [0, 0],
    }; // 9 = relu

    let buf_a = crate::transfer::src_buffer_tensor(a);

    // For unary, we safely pass `buf_a` as both inputs. The shader ignores `b`.
    compute_tensor_elementwise(session, buf_a, buf_a, &out_buffer, &params);

    crate::transfer::wrap_gpu_tensor(a.shape, a.strides, out_buffer, session.raw_device_arc())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_tensor::{CpuAllocator, Tensor};

    #[test]
    fn test_tensor_add_and_relu_e2e() {
        // Initialize the GPU Session
        let session = pollster::block_on(WgpuSession::new()).expect("Failed to init WgpuSession");

        // Create standard CPU Tensors
        let shape = [2, 2];

        // We will add these two.
        // 1.0 + 10.0 = 11.0
        // -20.0 + 5.0 = -15.0 (Should become 0.0 after ReLU)
        // 3.0 + 30.0 = 33.0
        // -50.0 + 10.0 = -40.0 (Should become 0.0 after ReLU)
        let data_a = vec![1.0f32, -20.0, 3.0, -50.0];
        let data_b = vec![10.0f32, 5.0, 30.0, 10.0];

        let cpu_a = Tensor::from_shape_vec(shape, data_a, CpuAllocator).unwrap();
        let cpu_b = Tensor::from_shape_vec(shape, data_b, CpuAllocator).unwrap();

        // Upload to GPU
        let gpu_a = session
            .upload_tensor(&cpu_a)
            .expect("Failed to upload tensor A");
        let gpu_b = session
            .upload_tensor(&cpu_b)
            .expect("Failed to upload tensor B");

        // Run Binary Math (Add)
        let gpu_added = add(&session, &gpu_a, &gpu_b).expect("Add operation failed");

        // Run Unary Math (ReLU) on the result of the Add
        let gpu_relu = relu(&session, &gpu_added).expect("ReLU operation failed");

        // Download results back to CPU
        let cpu_added = session
            .download_tensor(&gpu_added)
            .expect("Failed to download Add result");
        let cpu_relu = session
            .download_tensor(&gpu_relu)
            .expect("Failed to download ReLU result");

        // Verify the Add operation
        let added_slice = cpu_added.as_slice();
        assert_eq!(added_slice[0], 11.0);
        assert_eq!(added_slice[1], -15.0);
        assert_eq!(added_slice[2], 33.0);
        assert_eq!(added_slice[3], -40.0);

        // Verify the ReLU operation (Negatives should be 0.0)
        let relu_slice = cpu_relu.as_slice();
        assert_eq!(relu_slice[0], 11.0);
        assert_eq!(relu_slice[1], 0.0);
        assert_eq!(relu_slice[2], 33.0);
        assert_eq!(relu_slice[3], 0.0);
    }
}
