// ShaderKind is constructed with its bindings inline
//
// Adding a new op:
//   ShaderKind::new("MyOp", &[
//       binding!(0, Read,  "array<f32>", "input"),
//       binding!(1, Write, "array<f32>", "output"),
//   ])

// ── Binding description ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BindingRole {
    ReadBuffer,
    WriteBuffer,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BindingDesc {
    pub slot: u32,
    pub role: BindingRole,
    pub wgsl_type: &'static str,
    pub name: &'static str,
}

/// Shorthand for declaring a binding inline.
macro_rules! binding {
    ($slot:expr, Read, $ty:expr, $name:expr) => {
        BindingDesc {
            slot: $slot,
            role: BindingRole::ReadBuffer,
            wgsl_type: $ty,
            name: $name,
        }
    };
    ($slot:expr, Write, $ty:expr, $name:expr) => {
        BindingDesc {
            slot: $slot,
            role: BindingRole::WriteBuffer,
            wgsl_type: $ty,
            name: $name,
        }
    };
}

/// Identifies a compiled pipeline in the cache and carries its binding layout.
///
/// Two `ShaderKind`s are equal (and share a cached pipeline) when their
/// `name` strings are equal — the binding list is derived from the name
/// and is not compared independently.
#[derive(Debug, Clone)]
pub(crate) struct ShaderKind {
    /// Unique name used as the cache key and debug label.
    pub name: &'static str,
    /// Buffer bindings for this shader, declared at construction time.
    bindings: &'static [BindingDesc],
}

impl ShaderKind {
    pub(crate) const fn new(name: &'static str, bindings: &'static [BindingDesc]) -> Self {
        Self { name, bindings }
    }

    pub(crate) fn bindings(&self) -> &'static [BindingDesc] {
        self.bindings
    }
}

// Equality and hashing are based solely on the name string so that
// ShaderKind can be used as a HashMap key in the pipeline cache.
impl PartialEq for ShaderKind {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}
impl Eq for ShaderKind {}
impl std::hash::Hash for ShaderKind {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

// Declared as constants so ops can reference them without allocation.
// Adding a new op = add one constant here, nothing else in this file.

use std::sync::LazyLock;

macro_rules! shader_kind {
    ($name:ident, $key:literal, [$($bindings:expr),* $(,)?]) => {
        pub(crate) static $name: LazyLock<ShaderKind> = LazyLock::new(|| {
            static BINDINGS: &[BindingDesc] = &[$($bindings),*];
            ShaderKind::new($key, BINDINGS)
        });
    };
}

shader_kind!(
    RESIZE_NEAREST,
    "ResizeNearest",
    [
        binding!(0, Read, "array<f32>", "input_buf"),
        binding!(1, Write, "array<f32>", "output_buf"),
    ]
);

shader_kind!(
    RESIZE_BILINEAR,
    "ResizeBilinear",
    [
        binding!(0, Read, "array<f32>", "input_buf"),
        binding!(1, Write, "array<f32>", "output_buf"),
    ]
);

shader_kind!(
    TENSOR_ELEMENTWISE,
    "TensorElementwise",
    [
        binding!(0, Read, "array<vec4<f32>>", "a"),
        binding!(1, Read, "array<vec4<f32>>", "b"),
        binding!(2, Write, "array<vec4<f32>>", "out"),
    ]
);

shader_kind!(
    CAST_U8_TO_F32,
    "CastU8ToF32",
    [
        binding!(0, Read, "array<u32>", "input_u8"),
        binding!(1, Write, "array<f32>", "output_f32"),
    ]
);

// ── Pipeline cache key ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PipelineKey {
    pub shader_name: &'static str, // from ShaderKind::name
    pub pixel_bytes: u8,
    pub channels: u8,
    pub wg_x: u8,
    pub wg_y: u8,
    pub variant: u32,
}

impl PipelineKey {
    pub(crate) fn from_kind(
        kind: &ShaderKind,
        pixel_bytes: u8,
        channels: u8,
        wg_x: u8,
        wg_y: u8,
        variant: u32,
    ) -> Self {
        Self {
            shader_name: kind.name,
            pixel_bytes,
            channels,
            wg_x,
            wg_y,
            variant,
        }
    }
}

// ── WgslShader ────────────────────────────────────────────────────────────────

pub(crate) struct WgslShader {
    pub kind: ShaderKind,
    pub source: String,
}

impl WgslShader {
    /// Prepends `@group(0) @binding(N) var<storage, access> name: type;`
    /// declarations derived from the ShaderKind's binding list.
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
