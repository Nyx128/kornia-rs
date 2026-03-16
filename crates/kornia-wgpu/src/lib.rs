// src/lib.rs
pub mod allocator;
pub mod device;
pub mod error;
pub mod ops;
pub mod pixel;
pub mod session;
pub mod shader;
pub mod transfer;

pub use allocator::WgpuAllocator;
pub use error::WgpuError;
pub use pixel::GpuPixel;
pub use session::WgpuSession;

pub use transfer::{image_to_cpu, image_to_gpu};
