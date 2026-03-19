use crate::allocator::WgpuAllocator;
use crate::error::WgpuError;
use crate::gpu_res::{GpuImage, GpuTensor};
use crate::pixel::GpuPixel;
use crate::session::WgpuSession;
use kornia_image::allocator::{CpuAllocator, ImageAllocator};
use kornia_image::{Image, ImageSize};
use kornia_tensor::storage::TensorStorage;
use kornia_tensor::Tensor;

/// Wrap a `WgpuAllocator` into a `GpuImage`.
///
/// SAFETY: `TensorStorage::ptr` is `NonNull::dangling()`. The inner Image
/// must never have `.as_slice()` called on it — `GpuImage` enforces this
/// by not exposing that method.
///
/// # Arguments
///
/// * `size` - Dimensions of the resulting image.
/// * `alloc` - Pre-allocated WGPU memory.
///
/// # Returns
///
/// A wrapped [`GpuImage`].
///
/// # Errors
///
/// Currently always returns `Ok`, using `Result` for API consistency.
pub fn wrap_gpu_buffer<T: GpuPixel, const C: usize>(
    size: ImageSize,
    alloc: WgpuAllocator,
) -> Result<GpuImage<T, C>, WgpuError> {
    let numel = size.width * size.height * C;
    let storage = unsafe {
        TensorStorage::from_raw_parts(
            std::ptr::NonNull::<T>::dangling().as_ptr(),
            numel * std::mem::size_of::<T>(),
            alloc,
        )
    };
    Ok(GpuImage(Image(Tensor {
        storage,
        shape: [size.height, size.width, C],
        strides: [size.width * C, C, 1],
    })))
}

/// Upload a CPU image to the GPU via the compute pool.
///
/// # Arguments
///
/// * `session` - The active `WgpuSession`.
/// * `cpu_image` - The input image residing in CPU memory.
///
/// # Returns
///
/// A [`GpuImage`] whose data has been uploaded to VRAM.
///
/// # Errors
///
/// Passes through appropriate device errors up from mapping or similar issues.
pub fn image_to_gpu<T, const C: usize, A>(
    session: &WgpuSession,
    cpu_image: &Image<T, C, A>,
) -> Result<GpuImage<T, C>, WgpuError>
where
    T: GpuPixel,
    A: ImageAllocator,
{
    let size = cpu_image.size();
    let byte_size = (size.width * size.height * C * std::mem::size_of::<T>()) as u64;

    let alloc = session.acquire_compute(byte_size);
    session.raw_queue().write_buffer(
        alloc.gpu_buffer(),
        0,
        bytemuck::cast_slice(cpu_image.as_slice()),
    );

    wrap_gpu_buffer(size, alloc)
}

/// Download a GPU image to CPU RAM via the staging pool.
///
/// # Arguments
///
/// * `session` - The active `WgpuSession`.
/// * `gpu_image` - The input image residing in GPU memory.
///
/// # Returns
///
/// A generic `Image` mapped in CPU memory.
///
/// # Errors
///
/// Returns a [`WgpuError`] if mapping the underlying buffer fails.
pub fn image_to_cpu<T, const C: usize>(
    session: &WgpuSession,
    gpu_image: &GpuImage<T, C>,
) -> Result<Image<T, C, CpuAllocator>, WgpuError>
where
    T: GpuPixel + Clone,
{
    let size = gpu_image.size();
    let numel = size.width * size.height * C;
    let byte_size = (numel * std::mem::size_of::<T>()) as u64;

    let device = session.raw_device();
    let queue = session.raw_queue();
    let staging = session.staging_pool.acquire(device, byte_size);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("img-download"),
    });
    encoder.copy_buffer_to_buffer(
        gpu_image.0.storage.alloc().gpu_buffer(),
        0,
        &staging,
        0,
        byte_size,
    );
    let submit_idx = queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..byte_size);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submit_idx),
            timeout: None,
        })
        .unwrap();
    rx.recv().unwrap().map_err(WgpuError::MapFailed)?;

    let data: Vec<T> = bytemuck::cast_slice::<u8, T>(&slice.get_mapped_range()).to_vec();

    staging.unmap();
    session.staging_pool.release(staging);

    Image::new(size, data, CpuAllocator).map_err(|e| WgpuError::ImageError(e.to_string()))
}

// ── Internal helpers ──────────────────────────────────────────────────────────

#[inline]
pub(crate) fn src_buffer<T, const C: usize>(img: &GpuImage<T, C>) -> &wgpu::Buffer {
    img.0.storage.alloc().gpu_buffer()
}

#[inline]
pub(crate) fn src_buffer_tensor<T, const N: usize>(tensor: &GpuTensor<T, N>) -> &wgpu::Buffer {
    tensor.0.storage.alloc().gpu_buffer()
}

/// Wrap a `WgpuAllocator` into a `GpuTensor`.
pub(crate) fn wrap_gpu_tensor<T, const N: usize>(
    shape: [usize; N],
    strides: [usize; N],
    alloc: WgpuAllocator,
) -> Result<GpuTensor<T, N>, WgpuError> {
    let byte_size = shape.iter().product::<usize>() * std::mem::size_of::<T>();
    let storage = unsafe {
        TensorStorage::from_raw_parts(
            std::ptr::NonNull::<T>::dangling().as_ptr(),
            byte_size,
            alloc,
        )
    };
    Ok(GpuTensor(Tensor {
        storage,
        shape,
        strides,
    }))
}

/// Acquire a pooled output allocator for an op.
pub(crate) fn acquire_output(session: &WgpuSession, byte_size: u64) -> WgpuAllocator {
    session.acquire_compute(byte_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::{Image, ImageSize};
    use kornia_tensor::allocator::CpuAllocator;

    #[test]
    fn test_gpu_round_trip() {
        let session = pollster::block_on(WgpuSession::new()).unwrap();
        let size = ImageSize {
            width: 2,
            height: 2,
        };
        let original = Image::<u8, 1, _>::new(size, vec![10, 20, 30, 40], CpuAllocator).unwrap();

        let gpu = image_to_gpu(&session, &original).expect("upload failed");
        let downloaded = image_to_cpu(&session, &gpu).expect("download failed");

        assert_eq!(original.as_slice(), downloaded.as_slice());
    }
}
