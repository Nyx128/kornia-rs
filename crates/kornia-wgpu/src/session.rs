use crate::device::{DeviceOptions, WgpuDevice};
use crate::error::WgpuError;
use crate::ops::tensor::GpuElement;
use std::sync::Arc;

use crate::allocator::WgpuAllocator;
use kornia_tensor::allocator::CpuAllocator;
use kornia_tensor::Tensor;

#[derive(Clone)]
pub struct WgpuSession {
    pub(crate) device: Arc<WgpuDevice>,
}

impl WgpuSession {
    pub async fn new() -> Result<Self, WgpuError> {
        Self::with_options(DeviceOptions::default()).await
    }

    pub async fn with_options(options: DeviceOptions) -> Result<Self, WgpuError> {
        let device = WgpuDevice::new(options).await?;
        Ok(Self { device })
    }

    pub fn raw_device(&self) -> &wgpu::Device {
        &self.device.device
    }
    pub fn raw_queue(&self) -> &wgpu::Queue {
        &self.device.queue
    }

    pub(crate) fn raw_device_arc(&self) -> std::sync::Arc<WgpuDevice> {
        self.device.clone()
    }

    /// Uploads a CPU Tensor to the GPU
    pub fn upload_tensor<T, const N: usize>(
        &self,
        src: &Tensor<T, N, CpuAllocator>,
    ) -> Result<Tensor<T, N, WgpuAllocator>, WgpuError>
    where
        T: GpuElement,
    {
        let numel = src.shape.iter().product::<usize>();
        let byte_size = (numel * std::mem::size_of::<T>()) as wgpu::BufferAddress;

        let device_arc = self.raw_device_arc();
        let device = &device_arc.device;
        let queue = &device_arc.queue;

        let gpu_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tensor Upload Buffer"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Safely cast the CPU slice to bytes and write it to the GPU
        let cpu_slice = src.as_slice();
        let bytes = bytemuck::cast_slice(cpu_slice);
        queue.write_buffer(&gpu_buffer, 0, bytes);

        // Wrap it in our GPU allocator
        crate::transfer::wrap_gpu_tensor(
            src.shape.clone(),
            src.strides.clone(),
            gpu_buffer,
            device_arc.clone(),
        )
    }

    /// Downloads a GPU Tensor back to the CPU
    pub fn download_tensor<T, const N: usize>(
        &self,
        src: &Tensor<T, N, WgpuAllocator>,
    ) -> Result<Tensor<T, N, CpuAllocator>, WgpuError>
    where
        T: GpuElement + Clone,
    {
        let numel = src.shape.iter().product::<usize>();
        let byte_size = (numel * std::mem::size_of::<T>()) as wgpu::BufferAddress;

        let gpu_buffer = crate::transfer::src_buffer_tensor(src);

        let device_arc = self.raw_device_arc();
        let device = &device_arc.device;
        let queue = &device_arc.queue;

        // Create a staging buffer on the CPU side to read into
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tensor Download Staging"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy from the GPU storage buffer to the staging buffer
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(gpu_buffer, 0, &staging_buffer, 0, byte_size);
        queue.submit(std::iter::once(encoder.finish()));

        // Map the buffer so the CPU can read it
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().map_err(WgpuError::MapFailed)?;

        let mapped_view = buffer_slice.get_mapped_range();
        let typed_slice: &[T] = bytemuck::cast_slice(&mapped_view);
        let cpu_vec = typed_slice.to_vec();

        drop(mapped_view);
        staging_buffer.unmap();

        Ok(Tensor::from_shape_vec(src.shape.clone(), cpu_vec, CpuAllocator).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::WgpuAllocator;
    use kornia_tensor::TensorAllocator;
    use std::alloc::Layout;

    #[tokio::test]
    async fn test_session_creation() {
        let session = WgpuSession::new().await;
        assert!(session.is_ok(), "Failed to create WgpuSession");
    }

    #[tokio::test]
    async fn test_allocator_fails_loudly() {
        let session = WgpuSession::new().await.unwrap();

        // Create a dummy buffer just to test the allocator
        let buffer = session.raw_device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let cpu_backing = Arc::new(vec![0u8; 4 as usize]);

        let alloc = WgpuAllocator {
            device: session.device.clone(),
            gpu_buffer: std::sync::Arc::new(buffer),
            cpu_backing,
        };

        let layout = Layout::from_size_align(4, 1).unwrap();
        assert!(
            alloc.alloc(layout).is_err(),
            "Allocator should return an error"
        );
    }
}
