use crate::device::WgpuDevice;
use std::alloc::Layout;
use std::sync::Arc;

use kornia_image::allocator::ImageAllocator;
use kornia_tensor::{allocator::TensorAllocatorError, TensorAllocator};

// Add Debug to the derive macro
#[derive(Clone, Debug)]
pub struct WgpuAllocator {
    pub(crate) device: Arc<WgpuDevice>,
    pub(crate) gpu_buffer: Arc<wgpu::Buffer>,
    pub(crate) cpu_backing: Arc<Vec<u8>>
}

impl TensorAllocator for WgpuAllocator {
    fn alloc(&self, _layout: Layout) -> Result<*mut u8, TensorAllocatorError> {
        Err(TensorAllocatorError::NullPointer)
    }

    fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
        // No-op; handled by Arc<wgpu::Buffer> drop
    }
}

impl ImageAllocator for WgpuAllocator {}
