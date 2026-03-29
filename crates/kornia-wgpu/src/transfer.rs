use crate::allocator::WgpuAllocator;
use crate::error::WgpuError;
use crate::gpu_res::{GpuImage, GpuTensor};
use crate::pixel::GpuPixel;
use crate::session::WgpuSession;
use kornia_image::allocator::{CpuAllocator, ImageAllocator};
use kornia_image::{Image, ImageSize};
use kornia_tensor::storage::TensorStorage;
use kornia_tensor::Tensor;

/// wgpu requires all buffer copy sizes and offsets to be a multiple of 4.
const COPY_ALIGNMENT: u64 = wgpu::COPY_BUFFER_ALIGNMENT; // = 4

#[inline]
fn align_up(n: u64) -> u64 {
    (n + COPY_ALIGNMENT - 1) & !(COPY_ALIGNMENT - 1)
}

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
    let write_size = align_up(byte_size);

    // acquire_compute rounds up to next_power_of_two, so the buffer is always
    // at least `write_size` bytes. We pad the CPU slice with zero bytes when
    // the true payload is not a multiple of COPY_BUFFER_ALIGNMENT (4).
    let alloc = session.acquire_compute(write_size);
    let raw: &[u8] = bytemuck::cast_slice(cpu_image.as_slice());
    if write_size == byte_size {
        session.raw_queue().write_buffer(alloc.gpu_buffer(), 0, raw);
    } else {
        let mut padded = raw.to_vec();
        padded.resize(write_size as usize, 0u8);
        session.raw_queue().write_buffer(alloc.gpu_buffer(), 0, &padded);
    }

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

    // wgpu requires copy sizes to be a multiple of COPY_BUFFER_ALIGNMENT (4).
    // We request a staging buffer large enough for the aligned size, copy the
    // aligned amount, then slice only the true `byte_size` bytes when reading
    // back. The tail bytes in the padding region are defined (the pool buffer
    // is zeroed on first allocation by the driver) but are never exposed to
    // the caller.
    let copy_size = align_up(byte_size);

    let device = session.raw_device();
    let queue = session.raw_queue();
    let staging = session.staging_pool.acquire(device, copy_size);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("img-download"),
    });
    encoder.copy_buffer_to_buffer(
        gpu_image.0.storage.alloc().gpu_buffer(),
        0,
        &staging,
        0,
        copy_size, // aligned — never triggers COPY_BUFFER_ALIGNMENT validation error
    );
    let submit_idx = queue.submit(std::iter::once(encoder.finish()));

    // Map only the true byte range; the padding tail is invisible to the caller.
    let slice = staging.slice(..copy_size);
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

    // Cast the full mapped range, then truncate to the true element count.
    // This avoids a separate allocation: bytemuck::cast_slice is zero-copy,
    // and to_vec() copies only what we ask it to.
    let mapped = slice.get_mapped_range();
    let data: Vec<T> = bytemuck::cast_slice::<u8, T>(&mapped[..byte_size as usize]).to_vec();
    drop(mapped); // must drop before unmap

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

    /// Specifically exercises a non-multiple-of-4 byte size to catch
    /// COPY_BUFFER_ALIGNMENT regressions.
    #[test]
    fn test_gpu_round_trip_non_aligned() {
        let session = pollster::block_on(WgpuSession::new()).unwrap();

        // 3×1 u8 1-channel = 3 bytes — not aligned to 4
        let size = ImageSize {
            width: 3,
            height: 1,
        };
        let original =
            Image::<u8, 1, _>::new(size, vec![10u8, 20, 30], CpuAllocator).unwrap();
        let gpu = image_to_gpu(&session, &original).expect("upload failed");
        let downloaded = image_to_cpu(&session, &gpu).expect("download failed");
        assert_eq!(original.as_slice(), downloaded.as_slice());

        // 1×1 RGB f32 = 12 bytes — aligned, sanity check
        let size2 = ImageSize {
            width: 1,
            height: 1,
        };
        let original2 =
            Image::<f32, 3, _>::new(size2, vec![0.1f32, 0.5, 0.9], CpuAllocator).unwrap();
        let gpu2 = image_to_gpu(&session, &original2).expect("upload failed");
        let downloaded2 = image_to_cpu(&session, &gpu2).expect("download failed");
        assert_eq!(original2.as_slice(), downloaded2.as_slice());
    }
}