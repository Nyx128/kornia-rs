// src/transfer.rs
use crate::allocator::WgpuAllocator;
use crate::device::WgpuDevice;
use crate::error::WgpuError;
use crate::pixel::GpuPixel;
use crate::session::WgpuSession;
use kornia_image::allocator::{CpuAllocator, ImageAllocator};
use kornia_image::{Image, ImageSize}; // Adjust imports based on workspace
use kornia_tensor::storage::TensorStorage;
use kornia_tensor::Tensor;
use std::ptr::NonNull;
use std::sync::Arc;

pub(crate) fn wrap_gpu_buffer<T: GpuPixel, const C: usize>(
    size: ImageSize,
    buffer: wgpu::Buffer,
    device: Arc<WgpuDevice>,
) -> Result<Image<T, C, WgpuAllocator>, WgpuError> {
    let numel = size.width * size.height * C;
    let alloc = WgpuAllocator {
        device,
        buffer: Arc::new(buffer),
    };

    // SAFETY: ptr is NonNull::dangling() and is never dereferenced.
    unsafe {
        Image::from_raw_parts(size, NonNull::dangling().as_ptr(), numel, alloc)
            .map_err(|e| WgpuError::ImageError(e.to_string()))
    }
}

/// Uploads a CPU image to the GPU.
pub fn image_to_gpu<T, const C: usize, A>(
    session: &WgpuSession,
    cpu_image: &Image<T, C, A>,
) -> Result<Image<T, C, WgpuAllocator>, WgpuError>
where
    T: GpuPixel,
    A: ImageAllocator,
{
    let size = cpu_image.size();
    let numel = size.width * size.height * C;
    let byte_size = (numel * std::mem::size_of::<T>()) as wgpu::BufferAddress;

    // 1. Create a storage buffer on the GPU
    let buffer = session.raw_device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("GPU Image Buffer"),
        size: byte_size,
        // STORAGE for compute shaders, COPY_DST to write to it, COPY_SRC to read back
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // 2. Write the data to the buffer
    // bytemuck safely casts our strongly-typed pixel slice to raw bytes
    let byte_slice = bytemuck::cast_slice(cpu_image.as_slice());
    session.raw_queue().write_buffer(&buffer, 0, byte_slice);

    // 3. Wrap the new buffer in our custom allocator
    wrap_gpu_buffer(size, buffer, session.device.clone())
}

pub fn image_to_cpu<T, const C: usize>(
    session: &WgpuSession,
    gpu_image: &Image<T, C, WgpuAllocator>,
) -> Result<Image<T, C, CpuAllocator>, WgpuError>
where
    T: GpuPixel + Clone,
{
    let size = gpu_image.size();
    let numel = size.width * size.height * C;
    let byte_size = (numel * std::mem::size_of::<T>()) as wgpu::BufferAddress;

    let device = session.raw_device();
    let queue = session.raw_queue();

    // 1. Create a staging buffer
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Download Buffer"),
        size: byte_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // 2. Command the GPU to copy from our Image buffer to the Staging buffer
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Download Encoder"),
    });

    encoder.copy_buffer_to_buffer(
        &gpu_image.storage.alloc().buffer,
        0,
        &staging_buffer,
        0,
        byte_size,
    );

    // wgpu 28: submit() returns SubmissionIndex; pass it to poll for precise sync
    let submit_idx = queue.submit(std::iter::once(encoder.finish()));

    // 3. Map the staging buffer using standard library channels
    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();

    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });

    // 4. Block the current thread until the GPU finishes this specific submission
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submit_idx),
            timeout: None, // wait indefinitely
        })
        .unwrap();

    // Ensure mapping succeeded
    rx.recv().unwrap().map_err(WgpuError::MapFailed)?;

    // 5. Read the data, unmap, and construct the CPU image
    let data: Vec<T> = {
        let mapped_view = buffer_slice.get_mapped_range();
        bytemuck::cast_slice::<u8, T>(&mapped_view).to_vec()
    };

    staging_buffer.unmap();

    // Image::new generally defaults to CpuAllocator in Kornia
    Image::new(size, data, CpuAllocator).map_err(|e| WgpuError::ImageError(e.to_string()))
}

/// Helper to quickly extract the underlying wgpu::Buffer from a GPU image
#[inline]
pub(crate) fn src_buffer<T, const C: usize>(
    img: &kornia_image::Image<T, C, WgpuAllocator>,
) -> &wgpu::Buffer {
    &img.storage.alloc().buffer
}

/// Extracts the raw wgpu::Buffer reference from a GPU Tensor
pub(crate) fn src_buffer_tensor<T, const N: usize>(
    tensor: &Tensor<T, N, WgpuAllocator>,
) -> &wgpu::Buffer {
    // We use the `.alloc()` getter provided by TensorStorage
    // Assuming your WgpuAllocator struct has a public field named `buffer`
    &tensor.storage.alloc().buffer
}

/// Wraps a newly computed wgpu::Buffer into a full Kornia Tensor
pub(crate) fn wrap_gpu_tensor<T, const N: usize>(
    shape: [usize; N],
    strides: [usize; N],
    buffer: wgpu::Buffer,
    device_arc: Arc<WgpuDevice>, // or however your device arc is typed
) -> Result<Tensor<T, N, WgpuAllocator>, WgpuError> {
    let buffer_arc = Arc::new(buffer);
    let numel = shape.iter().product::<usize>();

    let byte_size = numel * std::mem::size_of::<T>();

    // 2. Instantiate your WgpuAllocator around the buffer
    // (Again, mirror your exact WgpuAllocator initialization from Image ops)
    let allocator = WgpuAllocator {
        device: device_arc,
        buffer: buffer_arc,
    };

    let dangling_ptr = std::ptr::NonNull::<T>::dangling().as_ptr();

    // Use the public constructor!
    let storage = unsafe { TensorStorage::from_raw_parts(dangling_ptr, byte_size, allocator) };

    // 4. Return the fully formed Tensor!
    Ok(Tensor {
        storage,
        shape,
        strides,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::{Image, ImageSize};
    use kornia_tensor::allocator::CpuAllocator; // Ensure this is imported

    #[test]
    fn test_gpu_round_trip() {
        let session = pollster::block_on(WgpuSession::new()).unwrap();

        let size = ImageSize {
            width: 2,
            height: 2,
        };
        let cpu_data: Vec<u8> = vec![10, 20, 30, 40];

        // Explicitly tell Rust this is a 1-channel u8 image
        let original_image = Image::<u8, 1, _>::new(size, cpu_data, CpuAllocator).unwrap();

        let gpu_image = image_to_gpu(&session, &original_image).expect("Failed to upload to GPU");

        let downloaded_image =
            image_to_cpu(&session, &gpu_image).expect("Failed to download to CPU");

        assert_eq!(original_image.as_slice(), downloaded_image.as_slice());
    }
}
