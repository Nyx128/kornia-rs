pub mod elementwise;

/// A trait for tensor element types that can be processed on the GPU.
pub trait GpuElement: bytemuck::Pod + bytemuck::Zeroable + Copy + Send + Sync + 'static {
    /// The WGSL type corresponding to this element type.
    const WGSL_TYPE: &'static str;
    /// The size of this element type in bytes.
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
