//! GPU acceleration for Kornia using [`wgpu`].
//!
//! This crate provides GPU-backed image and tensor types, as well as a collection
//! of operations that can be executed on the GPU using WGSL compute shaders.

pub mod allocator;
pub mod device;
pub mod error;
pub mod gpu_res;
pub mod ops;
pub mod pixel;
pub mod pool;
pub mod session;
pub mod shader;
pub mod transfer;
pub use gpu_res::{GpuImage, GpuTensor};

pub use allocator::WgpuAllocator;
pub use error::WgpuError;
pub use pixel::GpuPixel;
pub use pool::{BufferPool, StagingPoolMap};
pub use session::WgpuSession;
pub use transfer::{image_to_cpu, image_to_gpu};
