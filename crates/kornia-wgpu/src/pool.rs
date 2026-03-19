use std::collections::HashMap;
use std::sync::Mutex;

/// Reusable GPU compute (STORAGE) buffers, keyed by size class.
///
/// acquire() before each op, wrap in WgpuAllocator.
/// release() is called when the Arc<wgpu::Buffer> inside the
/// WgpuAllocator is dropped (i.e. when the output Image drops).
pub struct BufferPool {
    buckets: Mutex<HashMap<u64, Vec<wgpu::Buffer>>>,
    max_per_class: usize,
}

impl BufferPool {
    /// Creates a new `BufferPool`.
    ///
    /// # Arguments
    ///
    /// * `max_per_class` - Maximum number of unused buffers to keep per size class.
    ///
    /// # Returns
    ///
    /// A new [`BufferPool`].
    pub fn new(max_per_class: usize) -> Self {
        Self {
            buckets: Mutex::new(HashMap::new()),
            max_per_class,
        }
    }

    /// Acquires a buffer of at least `byte_size`.
    ///
    /// # Arguments
    ///
    /// * `device` - The `wgpu::Device` to create a new buffer on if needed.
    /// * `byte_size` - Minimum required buffer size in bytes.
    ///
    /// # Returns
    ///
    /// A [`wgpu::Buffer`] suitable for storage and copying.
    pub fn acquire(&self, device: &wgpu::Device, byte_size: u64) -> wgpu::Buffer {
        let class = byte_size.next_power_of_two().max(64);
        let recycled = self
            .buckets
            .lock()
            .unwrap()
            .get_mut(&class)
            .and_then(|v| v.pop());

        recycled.unwrap_or_else(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("pooled-compute-{class}")),
                size: class,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
    }

    /// Releases a buffer back into the pool.
    ///
    /// Buffer MUST be fully done on GPU before calling this.
    /// In practice this is guaranteed: release() is only reachable via
    /// Arc drop on WgpuAllocator, which only drops when the caller
    /// discards the output Image/Tensor.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The [`wgpu::Buffer`] to release.
    pub fn release(&self, buffer: wgpu::Buffer) {
        let class = buffer.size().next_power_of_two().max(64);
        let mut buckets = self.buckets.lock().unwrap();
        let bucket = buckets.entry(class).or_default();
        if bucket.len() < self.max_per_class {
            bucket.push(buffer);
        }
        // else: drop — wgpu frees VRAM automatically
    }

    /// Clears all cached buffers from the pool.
    pub fn clear(&self) {
        self.buckets.lock().unwrap().clear();
    }

    /// Returns the total number of cached buffers.
    ///
    /// # Returns
    ///
    /// The count of cached buffers.
    pub fn cached_count(&self) -> usize {
        self.buckets.lock().unwrap().values().map(|v| v.len()).sum()
    }
}

/// Pre-allocated MAP_READ staging buffers, one pool per size class.
/// Returned to pool immediately after unmap in image_to_cpu / download_tensor.
struct StagingPool {
    buffers: Mutex<Vec<wgpu::Buffer>>,
    max_count: usize,
}

impl StagingPool {
    fn new(device: &wgpu::Device, byte_size: u64, count: usize) -> Self {
        let buffers = (0..count)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("staging-{byte_size}-{i}")),
                    size: byte_size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        Self {
            buffers: Mutex::new(buffers),
            max_count: count,
        }
    }

    fn try_acquire(&self) -> Option<wgpu::Buffer> {
        self.buffers.lock().unwrap().pop()
    }

    fn release(&self, buf: wgpu::Buffer) {
        let mut bufs = self.buffers.lock().unwrap();
        if bufs.len() < self.max_count {
            bufs.push(buf);
        }
        // else drop
    }
}

/// Map of StagingPools keyed by size class. Grows on demand.
pub struct StagingPoolMap {
    pools: Mutex<HashMap<u64, StagingPool>>,
    count_per_class: usize,
}

impl StagingPoolMap {
    /// Creates a new `StagingPoolMap`.
    ///
    /// # Arguments
    ///
    /// * `count_per_class` - Maximum number of unused staging buffers to keep per size class.
    ///
    /// # Returns
    ///
    /// A new [`StagingPoolMap`].
    pub fn new(count_per_class: usize) -> Self {
        Self {
            pools: Mutex::new(HashMap::new()),
            count_per_class,
        }
    }

    /// Acquires a staging buffer of at least `byte_size`.
    ///
    /// # Arguments
    ///
    /// * `device` - The `wgpu::Device` to create a new buffer on if needed.
    /// * `byte_size` - Minimum required buffer size in bytes.
    ///
    /// # Returns
    ///
    /// A [`wgpu::Buffer`] suitable for mapping and copying.
    pub fn acquire(&self, device: &wgpu::Device, byte_size: u64) -> wgpu::Buffer {
        let class = byte_size.next_power_of_two().max(64);

        // fast path
        {
            let pools = self.pools.lock().unwrap();
            if let Some(pool) = pools.get(&class) {
                if let Some(buf) = pool.try_acquire() {
                    return buf;
                }
            }
        }

        // ensure pool exists for future release()
        self.pools
            .lock()
            .unwrap()
            .entry(class)
            .or_insert_with(|| StagingPool::new(device, class, self.count_per_class));

        // pool was empty — allocate overflow buffer
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("staging-overflow-{class}")),
            size: class,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Releases a staging buffer back into its pool.
    ///
    /// Buffer must be unmapped before calling this.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The [`wgpu::Buffer`] to release.
    pub fn release(&self, buffer: wgpu::Buffer) {
        let class = buffer.size();
        if let Some(pool) = self.pools.lock().unwrap().get(&class) {
            pool.release(buffer);
        }
        // else: no pool for this class (orphan overflow), just drop
    }

    /// Returns the total number of cached staging buffers.
    ///
    /// # Returns
    ///
    /// The count of cached staging buffers.
    pub fn cached_count(&self) -> usize {
        self.pools
            .lock()
            .unwrap()
            .values()
            .map(|p| p.buffers.lock().unwrap().len())
            .sum()
    }
}
