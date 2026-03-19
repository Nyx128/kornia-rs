//! Operations executable on the GPU (e.g., resizing, casting, elementwise ops).

pub mod image;
pub mod tensor;

use crate::session::WgpuSession;
use crate::shader::{PipelineKey, ShaderKind, WgslShader};

/// A generic dispatcher for operations that take 1 input buffer and write to 1 output buffer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_1in_1out<I: bytemuck::Pod>(
    session: &WgpuSession,
    shader_kind: &'static ShaderKind,
    shader_source: &'static str,
    in_buffer: &wgpu::Buffer,
    out_buffer: &wgpu::Buffer,
    immediates: &I,
    dispatch_size: (u32, u32),
    workgroup_size: (u8, u8),
    pixel_bytes: u8,
    channels: u8,
) {
    let device_arc = session.raw_device_arc();
    let device = &device_arc.device;
    let queue = &device_arc.queue;

    let key = PipelineKey::from_kind(
        shader_kind,
        pixel_bytes,
        channels,
        workgroup_size.0,
        workgroup_size.1,
        0,
    );

    let mut shader = WgslShader {
        kind: shader_kind.clone(),
        source: shader_source.to_string(),
    };
    shader.build();

    // No bgl_entries argument — device derives them from shader.kind.bindings()
    let pipeline = device_arc.get_or_create_pipeline(key, &shader, std::mem::size_of::<I>() as u32);

    // Build bind group entries from the pipeline's bind group layout
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("1in_1out Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: in_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
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
        cpass.set_immediates(0, bytemuck::bytes_of(immediates));

        let wg_x = dispatch_size.0.div_ceil(workgroup_size.0 as u32);
        let wg_y = dispatch_size.1.div_ceil(workgroup_size.1 as u32);
        cpass.dispatch_workgroups(wg_x, wg_y, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
}
