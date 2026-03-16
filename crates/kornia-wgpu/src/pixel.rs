/// Pixel types that can be processed on the GPU.
pub trait GpuPixel: bytemuck::Pod + bytemuck::Zeroable + Copy + Send + Sync + 'static {
    const WGSL_TYPE: &'static str;
    const BYTE_SIZE: usize = std::mem::size_of::<Self>();
    const PACK_FACTOR: usize;
}

impl GpuPixel for u8 {
    const WGSL_TYPE: &'static str = "u32";
    const PACK_FACTOR: usize = 4;
}
impl GpuPixel for u16 {
    const WGSL_TYPE: &'static str = "u32";
    const PACK_FACTOR: usize = 2;
}
impl GpuPixel for f32 {
    const WGSL_TYPE: &'static str = "f32";
    const PACK_FACTOR: usize = 1;
}
