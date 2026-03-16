#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum ShaderKind {
    ResizeNearest,
    ResizeBilinear,
    // ResizeBilinear, GaussianBlur, etc.)

    //tensor ops
    TensorElementwise, // For simple elementwise ops like add, mul, etc.
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BindingRole {
    ReadBuffer,
    WriteBuffer,
}
pub(crate) struct BindingDesc {
    pub slot: u32,
    pub role: BindingRole,
    pub wgsl_type: &'static str, // "array<f32>", "array<u32>", etc.
    pub name: &'static str,      // "input_buf", "output_buf", "buf_a", etc.
}

impl ShaderKind {
    pub(crate) fn bindings(&self) -> &'static [BindingDesc] {
        match self {
            ShaderKind::ResizeNearest | ShaderKind::ResizeBilinear => &[
                BindingDesc {
                    slot: 0,
                    role: BindingRole::ReadBuffer,
                    wgsl_type: "array<f32>",
                    name: "input_buf",
                },
                BindingDesc {
                    slot: 1,
                    role: BindingRole::WriteBuffer,
                    wgsl_type: "array<f32>",
                    name: "output_buf",
                },
            ],
            ShaderKind::TensorElementwise => &[
                BindingDesc {
                    slot: 0,
                    role: BindingRole::ReadBuffer,
                    wgsl_type: "array<f32>",
                    name: "a",
                },
                BindingDesc {
                    slot: 1,
                    role: BindingRole::ReadBuffer,
                    wgsl_type: "array<f32>",
                    name: "b",
                },
                BindingDesc {
                    slot: 2,
                    role: BindingRole::WriteBuffer,
                    wgsl_type: "array<f32>",
                    name: "out",
                },
            ],
        }
    }
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
        let prefix: String = self
            .kind
            .bindings()
            .iter()
            .map(|b| {
                let access = match b.role {
                    BindingRole::ReadBuffer => "read",
                    BindingRole::WriteBuffer => "read_write",
                };
                format!(
                    "@group(0) @binding({}) var<storage, {}> {}: {};\n",
                    b.slot, access, b.name, b.wgsl_type
                )
            })
            .collect();

        self.source = format!("{}\n{}", prefix, self.source);
    }
}
