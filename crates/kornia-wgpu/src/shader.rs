// src/shader.rs
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum ShaderKind {
    ResizeNearest,
    ResizeBilinear,
    // add more here later (ResizeBilinear, GaussianBlur, etc.)

    //tensor ops
    TensorElementwise, // For simple elementwise ops like add, mul, etc.
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PipelineKey {
    pub shader_id: ShaderKind,
    pub pixel_bytes: u8,
    pub channels: u8,
    pub wg_x: u8,
    pub wg_y: u8,
    pub variant: u32, // For any extra operation-specific flags
}

pub(crate) struct WgslShader {
    pub kind: ShaderKind,
    pub source: String,
}

impl WgslShader {
    pub fn build(&mut self) {
        let prefix = r#"
            @group(0) @binding(0) var<storage, read> input_buf: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output_buf: array<f32>;
        "#;

        // Overwrite our own source with the fully combined string
        self.source = format!("{}\n{}", prefix, self.source);
    }
}
