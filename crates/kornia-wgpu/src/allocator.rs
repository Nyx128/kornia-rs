use crate::error::WgpuError;
use crate::pool::BufferPool;
use std::alloc::Layout;
use std::sync::Arc;

use kornia_image::allocator::ImageAllocator;
use kornia_tensor::{allocator::TensorAllocatorError, TensorAllocator};

/// RAII guard that returns a `wgpu::Buffer` to its `BufferPool` on drop.
///
/// `WgpuAllocator` holds one of these (wrapped in Arc) instead of a bare
/// `Arc<wgpu::Buffer>`. When the last clone of the allocator drops, this
/// guard fires and puts the buffer back into the pool — closing the reuse loop.
pub(crate) struct PooledBufferGuard {
    pub(crate) buffer: Option<wgpu::Buffer>,
    pub(crate) pool: Option<Arc<BufferPool>>,
}

impl PooledBufferGuard {
    /// Buffer that belongs to a pool and is returned to it on drop.
    pub fn pooled(buffer: wgpu::Buffer, pool: Arc<BufferPool>) -> Self {
        Self {
            buffer: Some(buffer),
            pool: Some(pool),
        }
    }

    #[inline]
    pub fn buffer(&self) -> &wgpu::Buffer {
        self.buffer
            .as_ref()
            .expect("PooledBufferGuard already consumed")
    }
}

impl Drop for PooledBufferGuard {
    fn drop(&mut self) {
        match (self.buffer.take(), self.pool.take()) {
            (Some(buf), Some(pool)) => pool.release(buf),
            _ => { /* buffer drops normally */ }
        }
    }
}

impl std::fmt::Debug for PooledBufferGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PooledBufferGuard")
            .field("has_buffer", &self.buffer.is_some())
            .field("pooled", &self.pool.is_some())
            .finish()
    }
}

/// GPU allocator backing kornia `Image` and `Tensor` types.
///
/// # Invariant — the CPU pointer is always dangling
///
/// The `*mut T` stored in `TensorStorage` is `NonNull::dangling()`.
/// It must never be dereferenced. All data access goes through
/// `WgpuAllocator::gpu_buffer()` via the transfer functions.
#[derive(Clone, Debug)]
pub struct WgpuAllocator {
    /// Shared guard. All clones of this allocator reference the same buffer.
    /// Pool release fires exactly once when the last clone drops.
    pub(crate) guard: Arc<PooledBufferGuard>,
}

impl WgpuAllocator {
    /// Returns a reference to the underlying WGPU buffer.
    ///
    /// # Returns
    ///
    /// A reference to the [`wgpu::Buffer`].
    #[inline]
    pub fn gpu_buffer(&self) -> &wgpu::Buffer {
        self.guard.buffer()
    }

    /// Returns the standard error for invalid CPU access to GPU memory.
    ///
    /// # Returns
    ///
    /// A [`WgpuError::CpuAccessOnGpuBuffer`].
    #[cold]
    #[inline(never)]
    pub fn cpu_access_error() -> WgpuError {
        WgpuError::CpuAccessOnGpuBuffer
    }

    /// Helper to return a CPU access error for a generic type `T`.
    ///
    /// # Returns
    ///
    /// An `Err` containing the standard CPU access error.
    ///
    /// # Errors
    ///
    /// Always returns [`WgpuError::CpuAccessOnGpuBuffer`].
    #[inline]
    pub fn err_cpu_access<T>() -> Result<T, WgpuError> {
        Err(Self::cpu_access_error())
    }
}

impl TensorAllocator for WgpuAllocator {
    fn alloc(&self, _layout: Layout) -> Result<*mut u8, TensorAllocatorError> {
        Err(TensorAllocatorError::NullPointer)
    }

    fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
        // No-op: Arc<PooledBufferGuard>::drop handles both VRAM cleanup
        // and pool return when the last WgpuAllocator clone is dropped.
    }
}

impl ImageAllocator for WgpuAllocator {}
