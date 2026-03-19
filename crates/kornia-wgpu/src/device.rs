use crate::error::WgpuError;
use crate::shader::BindingRole;
use crate::shader::{PipelineKey, WgslShader};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Configuration options for creating a [`WgpuDevice`].
pub struct DeviceOptions {
    pub backend: Option<wgpu::Backends>,
    pub power_preference: wgpu::PowerPreference,
    pub workgroup_size: (u32, u32),
    pub staging_pool_size: usize,
}

impl Default for DeviceOptions {
    fn default() -> Self {
        Self {
            backend: None,
            power_preference: wgpu::PowerPreference::HighPerformance,
            workgroup_size: (8, 8),
            staging_pool_size: 64 * 1024 * 1024,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct WgpuDevice {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub limits: wgpu::Limits,
    pub features: wgpu::Features,
    pub workgroup_size: (u32, u32),
    pipeline_cache: Mutex<HashMap<PipelineKey, Arc<wgpu::ComputePipeline>>>,
    // staging_pool: Mutex<StagingPool>, will be considered later on
}

impl WgpuDevice {
    pub(crate) async fn new(options: DeviceOptions) -> Result<Arc<Self>, WgpuError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: options
                .backend
                .unwrap_or(wgpu::Backends::VULKAN | wgpu::Backends::DX12 | wgpu::Backends::METAL),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: options.power_preference,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|_| WgpuError::NoAdapter(options.backend))?;

        let adapter_limits = adapter.limits();

        println!("Selected adapter: {:?}", adapter.get_info().name);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::IMMEDIATES,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                required_limits: adapter_limits,
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await?;

        Ok(Arc::new(Self {
            device,
            queue,
            limits: adapter.limits(),
            features: adapter.features(),
            workgroup_size: options.workgroup_size,
            pipeline_cache: Mutex::new(HashMap::new()),
        }))
    }

    pub(crate) fn get_or_create_pipeline(
        self: &Arc<Self>,
        key: PipelineKey,
        shader: &WgslShader,
        immediate_size: u32,
    ) -> Arc<wgpu::ComputePipeline> {
        // Fast path: Check the cache first
        {
            let cache = self.pipeline_cache.lock().unwrap();
            if let Some(pipeline) = cache.get(&key) {
                return pipeline.clone();
            }
        }

        // Slow path: Compile the shader
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("{:?}", key.shader_name)),
                source: wgpu::ShaderSource::Wgsl(shader.source.as_str().into()),
            });

        let bgl_entries: Vec<wgpu::BindGroupLayoutEntry> = shader
            .kind
            .bindings()
            .iter()
            .map(|b| wgpu::BindGroupLayoutEntry {
                binding: b.slot,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: match b.role {
                        BindingRole::ReadBuffer => {
                            wgpu::BufferBindingType::Storage { read_only: true }
                        }
                        BindingRole::WriteBuffer => {
                            wgpu::BufferBindingType::Storage { read_only: false }
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Compute BGL"),
                    entries: &bgl_entries,
                });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                immediate_size,
            });

        let pipeline = Arc::new(self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{:?} Pipeline", key.shader_name)),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        //Store in cache
        let mut cache = self.pipeline_cache.lock().unwrap();
        cache.insert(key, pipeline.clone());

        pipeline
    }
}
