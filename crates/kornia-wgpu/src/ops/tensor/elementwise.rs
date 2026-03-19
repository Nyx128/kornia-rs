use crate::error::WgpuError;
use crate::gpu_res::GpuTensor;
use crate::session::WgpuSession;
use crate::shader::{PipelineKey, WgslShader, TENSOR_ELEMENTWISE};
use crate::transfer::{acquire_output, src_buffer_tensor, wrap_gpu_tensor};

/// Interoperability parameters for elementwise shaders.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ElemParams {
    pub numel: u32,
    pub op_kind: u32,
    pub dispatch_x: u32,
    pub _pad: u32,
}

const ELEMENTWISE_WGSL: &str = r#"
struct Params {
    numel:   u32,  // original element count
    op_kind: u32,
    dispatch_x:u32,
    _pad:    u32,
}
var<immediate> p: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * p.dispatch_x * 256u;

    // Each thread handles 4 elements. Stop when we've covered all vec4 slots.
    // numel is the original count — div_ceil(numel, 4) is the number of vec4s.
    let vec4_count = (p.numel + 3u) / 4u;
    if i >= vec4_count { return; }

    let av = a[i];
    let bv = b[i];
    var res: vec4<f32>;

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
        case 9u: { res = max(vec4(0.0), av); }
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
    numel: u32,
    op_kind: u32,
) {
    let device_arc = session.raw_device_arc();
    let device = &device_arc.device;
    let queue = &device_arc.queue;

    let key = PipelineKey::from_kind(&TENSOR_ELEMENTWISE.clone(), 4, 1, 64, 1, 0);

    let mut shader = WgslShader {
        kind: TENSOR_ELEMENTWISE.clone(),
        source: ELEMENTWISE_WGSL.to_string(),
    };
    shader.build();

    let (wg_x, wg_y) = compute_dispatch(numel);

    let params = ElemParams {
        numel,
        dispatch_x: wg_x,
        op_kind,
        _pad: 0,
    };

    let pipeline =
        device_arc.get_or_create_pipeline(key, &shader, std::mem::size_of::<ElemParams>() as u32);

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("elementwise"),
        layout: &pipeline.get_bind_group_layout(0),
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
        cpass.set_immediates(0, bytemuck::bytes_of(&params));
        cpass.dispatch_workgroups(wg_x, wg_y, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
}

fn compute_dispatch(numel: u32) -> (u32, u32) {
    const MAX_WG: u32 = 65535;
    let vec4_count = numel.div_ceil(4);
    let total_wg = vec4_count.div_ceil(256);

    if total_wg <= MAX_WG {
        (total_wg, 1)
    } else {
        // Spill into Y: find smallest Y such that X <= 65535
        let wg_y = total_wg.div_ceil(MAX_WG);
        let wg_x = total_wg.div_ceil(wg_y);
        (wg_x, wg_y)
    }
}

/// Adds two GPU tensors elementwise.
///
/// # Arguments
///
/// * `session` - The active `WgpuSession`.
/// * `a` - The first input tensor.
/// * `b` - The second input tensor.
///
/// # Returns
///
/// A new [`GpuTensor`] containing the elementwise sum.
///
/// # Errors
///
/// Can return a [`WgpuError`] if buffer allocation fails.
pub fn add<const N: usize>(
    session: &WgpuSession,
    a: &GpuTensor<f32, N>,
    b: &GpuTensor<f32, N>,
) -> Result<GpuTensor<f32, N>, WgpuError> {
    let numel = a.shape().iter().product::<usize>() as u32;
    let byte_size = (numel as u64) * 4;
    let byte_size_pad = (byte_size + 15) & !15; // align to vec4
    let out_alloc = acquire_output(session, byte_size_pad);

    compute_tensor_elementwise(
        session,
        src_buffer_tensor(a),
        src_buffer_tensor(b),
        out_alloc.gpu_buffer(),
        numel,
        0, // op_kind 0 = add
    );

    wrap_gpu_tensor(a.0.shape, a.0.strides, out_alloc)
}

/// Applies the Rectified Linear Unit (ReLU) activation function elementwise.
///
/// # Arguments
///
/// * `session` - The active `WgpuSession`.
/// * `a` - The input tensor.
///
/// # Returns
///
/// A new [`GpuTensor`] containing the activated values.
///
/// # Errors
///
/// Can return a [`WgpuError`] if buffer allocation fails.
pub fn relu<const N: usize>(
    session: &WgpuSession,
    a: &GpuTensor<f32, N>,
) -> Result<GpuTensor<f32, N>, WgpuError> {
    let numel = a.shape().iter().product::<usize>() as u32;
    let byte_size = (numel as u64) * 4;
    let byte_size_pad = (byte_size + 15) & !15;
    let out_alloc = acquire_output(session, byte_size_pad);

    compute_tensor_elementwise(
        session,
        src_buffer_tensor(a),
        src_buffer_tensor(a),
        out_alloc.gpu_buffer(),
        numel,
        9, // op_kind 9 = relu
    );

    wrap_gpu_tensor(a.0.shape, a.0.strides, out_alloc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_tensor::{CpuAllocator, Tensor};

    #[test]
    fn test_tensor_add_and_relu_e2e() {
        let session = pollster::block_on(WgpuSession::new()).unwrap();

        let shape = [2, 2];
        let cpu_a =
            Tensor::from_shape_vec(shape, vec![1.0f32, -20.0, 3.0, -50.0], CpuAllocator).unwrap();
        let cpu_b =
            Tensor::from_shape_vec(shape, vec![10.0f32, 5.0, 30.0, 10.0], CpuAllocator).unwrap();

        let gpu_a = session.upload_tensor(&cpu_a).unwrap();
        let gpu_b = session.upload_tensor(&cpu_b).unwrap();
        let gpu_sum = add(&session, &gpu_a, &gpu_b).unwrap();
        let gpu_relu = relu(&session, &gpu_sum).unwrap();

        let cpu_sum = session.download_tensor(&gpu_sum).unwrap();
        let cpu_relu = session.download_tensor(&gpu_relu).unwrap();

        assert_eq!(cpu_sum.as_slice(), &[11.0, -15.0, 33.0, -40.0]);
        assert_eq!(cpu_relu.as_slice(), &[11.0, 0.0, 33.0, 0.0]);
    }
}
