use crate::allocator::{PooledBufferGuard, WgpuAllocator};
use crate::device::{DeviceOptions, WgpuDevice};
use crate::error::WgpuError;
use crate::gpu_res::GpuTensor;
use crate::ops::tensor::GpuElement;
use crate::pool::{BufferPool, StagingPoolMap};
use kornia_tensor::allocator::CpuAllocator;
use kornia_tensor::Tensor;
use std::sync::Arc;

/// A session for executing compute operations on the GPU.
///
/// This holds the core device, queue, and resource pools needed to
/// run operations and manage memory efficiently.
#[derive(Clone)]
pub struct WgpuSession {
    pub(crate) device: Arc<WgpuDevice>,
    pub(crate) compute_pool: Arc<BufferPool>,
    pub(crate) staging_pool: Arc<StagingPoolMap>,
}

impl WgpuSession {
    /// Creates a new `WgpuSession` with default options.
    ///
    /// # Returns
    ///
    /// A new [`WgpuSession`] or an error if initialization fails.
    ///
    /// # Errors
    ///
    /// Returns a [`WgpuError`] if the GPU device could not be acquired.
    pub async fn new() -> Result<Self, WgpuError> {
        Self::with_options(DeviceOptions::default()).await
    }

    /// Creates a new `WgpuSession` with the specified options.
    ///
    /// # Arguments
    ///
    /// * `options` - Configuration options for device creation.
    ///
    /// # Returns
    ///
    /// A new [`WgpuSession`] or an error if initialization fails.
    ///
    /// # Errors
    ///
    /// Returns a [`WgpuError`] if the GPU device could not be acquired.
    pub async fn with_options(options: DeviceOptions) -> Result<Self, WgpuError> {
        let device = WgpuDevice::new(options).await?;
        Ok(Self {
            device,
            compute_pool: Arc::new(BufferPool::new(4)),
            staging_pool: Arc::new(StagingPoolMap::new(3)),
        })
    }

    /// Returns a reference to the raw `wgpu::Device`.
    ///
    /// # Returns
    ///
    /// A reference to the [`wgpu::Device`].
    pub fn raw_device(&self) -> &wgpu::Device {
        &self.device.device
    }
    /// Returns a reference to the raw `wgpu::Queue`.
    ///
    /// # Returns
    ///
    /// A reference to the [`wgpu::Queue`].
    pub fn raw_queue(&self) -> &wgpu::Queue {
        &self.device.queue
    }
    pub(crate) fn raw_device_arc(&self) -> Arc<WgpuDevice> {
        self.device.clone()
    }

    /// Acquire a pooled compute buffer and wrap it in a `WgpuAllocator`.
    ///
    /// This is the single entry point for all compute buffer allocation.
    /// `transfer::image_to_gpu`, `upload_tensor`, and every op output go
    /// through here. The buffer is returned to the pool automatically when
    /// the last clone of the returned `WgpuAllocator` is dropped.
    pub(crate) fn acquire_compute(&self, byte_size: u64) -> WgpuAllocator {
        let buffer = self.compute_pool.acquire(self.raw_device(), byte_size);
        WgpuAllocator {
            guard: Arc::new(PooledBufferGuard::pooled(buffer, self.compute_pool.clone())),
        }
    }

    /// Upload a CPU tensor to the GPU via the compute pool.
    ///
    /// # Arguments
    ///
    /// * `src` - The CPU tensor to upload.
    ///
    /// # Returns
    ///
    /// A new [`GpuTensor`] residing on the GPU.
    ///
    /// # Errors
    ///
    /// Can return a [`WgpuError`] if buffer upload fails.
    pub fn upload_tensor<T, const N: usize>(
        &self,
        src: &Tensor<T, N, CpuAllocator>,
    ) -> Result<GpuTensor<T, N>, WgpuError>
    where
        T: GpuElement,
    {
        let numel = src.shape.iter().product::<usize>();
        let byte_size = (numel * std::mem::size_of::<T>()) as u64;

        let alloc = self.acquire_compute(byte_size);
        self.raw_queue()
            .write_buffer(alloc.gpu_buffer(), 0, bytemuck::cast_slice(src.as_slice()));

        crate::transfer::wrap_gpu_tensor(src.shape, src.strides, alloc)
    }

    /// Download a GPU tensor to the CPU via the staging pool.
    ///
    /// # Arguments
    ///
    /// * `src` - The GPU tensor to download.
    ///
    /// # Returns
    ///
    /// A new CPU tensor containing the downloaded data.
    ///
    /// # Errors
    ///
    /// Returns a [`WgpuError`] if mapping the underlying buffer fails.
    pub fn download_tensor<T, const N: usize>(
        &self,
        src: &GpuTensor<T, N>,
    ) -> Result<Tensor<T, N, CpuAllocator>, WgpuError>
    where
        T: GpuElement + Clone,
    {
        let numel = src.shape().iter().product::<usize>();
        let byte_size = (numel * std::mem::size_of::<T>()) as u64;

        let device = self.raw_device();
        let queue = self.raw_queue();
        let staging = self.staging_pool.acquire(device, byte_size);

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(
            crate::transfer::src_buffer_tensor(src),
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

        let cpu_vec: Vec<T> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();

        staging.unmap();
        self.staging_pool.release(staging);

        Ok(Tensor::from_shape_vec(src.0.shape, cpu_vec, CpuAllocator).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transfer::{image_to_cpu, image_to_gpu};
    use kornia_image::{allocator::CpuAllocator, Image, ImageSize};

    fn make_session() -> Option<WgpuSession> {
        pollster::block_on(WgpuSession::new()).ok()
    }

    fn test_image_4x4() -> Image<f32, 1, CpuAllocator> {
        let size = ImageSize {
            width: 4,
            height: 4,
        };
        Image::new(size, (0..16).map(|i| i as f32).collect(), CpuAllocator).unwrap()
    }

    #[test]
    fn compute_pool_recycles_buffer() {
        let Some(session) = make_session() else {
            return;
        };
        let cpu = test_image_4x4();

        let before = session.compute_pool.cached_count();

        let gpu1 = image_to_gpu(&session, &cpu).unwrap();
        assert_eq!(
            session.compute_pool.cached_count(),
            before,
            "cached_count must not increase while buffer is in use"
        );

        drop(gpu1);

        assert_eq!(
            session.compute_pool.cached_count(),
            before + 1,
            "cached_count must increase by 1 after Image is dropped"
        );

        let gpu2 = image_to_gpu(&session, &cpu).unwrap();
        assert_eq!(
            session.compute_pool.cached_count(),
            before,
            "cached_count must decrease by 1 when pool acquires again"
        );

        drop(gpu2);
    }

    #[test]
    fn compute_pool_separate_size_classes() {
        let Some(session) = make_session() else {
            return;
        };

        let small = Image::<f32, 1, _>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![1.0f32; 4],
            CpuAllocator,
        )
        .unwrap();
        let large = Image::<f32, 1, _>::new(
            ImageSize {
                width: 8,
                height: 8,
            },
            vec![1.0f32; 64],
            CpuAllocator,
        )
        .unwrap();

        let gpu_s = image_to_gpu(&session, &small).unwrap();
        let gpu_l = image_to_gpu(&session, &large).unwrap();

        let ptr_s = crate::transfer::src_buffer(&gpu_s) as *const wgpu::Buffer;
        let ptr_l = crate::transfer::src_buffer(&gpu_l) as *const wgpu::Buffer;

        assert_ne!(
            ptr_s, ptr_l,
            "different size classes must not share a buffer"
        );
    }

    #[test]
    fn staging_pool_recycles_and_data_is_correct() {
        let Some(session) = make_session() else {
            return;
        };
        let cpu = test_image_4x4();
        let gpu = image_to_gpu(&session, &cpu).unwrap();

        let dl1 = image_to_cpu(&session, &gpu).unwrap();
        assert_eq!(
            dl1.as_slice(),
            cpu.as_slice(),
            "first download must be exact"
        );

        assert!(
            session.staging_pool.cached_count() >= 1,
            "staging pool must have reclaimed the buffer after download"
        );

        let dl2 = image_to_cpu(&session, &gpu).unwrap();
        assert_eq!(
            dl2.as_slice(),
            cpu.as_slice(),
            "second download via recycled buffer must be exact"
        );
    }

    #[test]
    fn staging_pool_overflow_still_works() {
        let Some(session) = make_session() else {
            return;
        };
        let cpu = test_image_4x4();
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let gpu = image_to_gpu(&session, &cpu).unwrap();

        for i in 0..5 {
            let dl = image_to_cpu(&session, &gpu).unwrap();
            assert_eq!(
                dl.as_slice(),
                data.as_slice(),
                "download {i}: data corrupted"
            );
        }
    }

    #[test]
    fn compute_pool_respects_max_per_class() {
        let Some(session) = make_session() else {
            return;
        };
        let cpu = test_image_4x4();

        let images: Vec<_> = (0..6)
            .map(|_| image_to_gpu(&session, &cpu).unwrap())
            .collect();
        drop(images);

        assert_eq!(
            session.compute_pool.cached_count(),
            4,
            "pool must cap at max_per_class=4, not cache all 6"
        );
    }

    #[test]
    fn round_trip_preserves_pixels() {
        let Some(session) = make_session() else {
            return;
        };
        let numel = 32 * 32 * 3;
        let data: Vec<f32> = (0..numel).map(|i| i as f32 / numel as f32).collect();
        let cpu = Image::<f32, 3, _>::new(
            ImageSize {
                width: 32,
                height: 32,
            },
            data.clone(),
            CpuAllocator,
        )
        .unwrap();

        let gpu = image_to_gpu(&session, &cpu).unwrap();
        let dl = image_to_cpu(&session, &gpu).unwrap();

        assert_eq!(dl.as_slice().len(), numel);
        for (i, (got, expected)) in dl.as_slice().iter().zip(data.iter()).enumerate() {
            assert_eq!(got, expected, "pixel {i} mismatch");
        }
    }
}
