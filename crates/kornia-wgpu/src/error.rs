#[derive(thiserror::Error, Debug)]
pub enum WgpuError {
    #[error("No suitable GPU adapter found. Requested backend: {0:?}")]
    NoAdapter(Option<wgpu::Backends>),

    #[error("Device creation failed: {0}")]
    DeviceCreation(#[from] wgpu::RequestDeviceError),

    #[error("Buffer mapping failed: {0:?}")]
    MapFailed(wgpu::BufferAsyncError),

    // Mocking kornia_image::ImageError for now; replace with actual if needed
    #[error("Image error: {0}")]
    ImageError(String),

    #[error("Size mismatch: expected {expected} bytes, got {got}")]
    SizeMismatch { expected: usize, got: usize },

    #[error("Unsupported pixel type: {0}")]
    UnsupportedPixelType(&'static str),

    #[error("Invalid kernel size {0}: must be odd and >= 1")]
    InvalidKernelSize(usize),

    #[error("Invalid sigma {0}: must be > 0")]
    InvalidSigma(f32),

    #[error("Shader compilation failed for {shader:?}: {msg}")]
    ShaderCompilation { shader: String, msg: String }, // Using String for shader kind temporarily

    #[error("Custom dispatch error: {0}")]
    Custom(String),
}
