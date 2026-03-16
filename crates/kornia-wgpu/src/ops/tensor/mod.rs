pub mod elementwise;

pub trait GpuElement: bytemuck::Pod + bytemuck::Zeroable + Copy + Send + Sync + 'static {
    const WGSL_TYPE: &'static str;
    const BYTE_SIZE: usize = std::mem::size_of::<Self>();
}

impl GpuElement for f32 {
    const WGSL_TYPE: &'static str = "f32";
}
impl GpuElement for u32 {
    const WGSL_TYPE: &'static str = "u32";
}
impl GpuElement for i32 {
    const WGSL_TYPE: &'static str = "i32";
}
