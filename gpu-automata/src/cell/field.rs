// ── cell/field.rs ─────────────────────────────────────────────────────────────
//
// A cell field has a name, a WGSL-compatible scalar type, and an optional
// default value that is used when initialising the GPU buffer.

/// Scalar types supported inside a cell struct.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldType {
    /// 32-bit unsigned integer  (`u32` in WGSL).
    U32,
    /// 32-bit IEEE float (`f32` in WGSL).
    F32,
}

impl FieldType {
    /// Returns the WGSL type keyword.
    pub fn wgsl_type(self) -> &'static str {
        match self {
            FieldType::U32 => "u32",
            FieldType::F32 => "f32",
        }
    }

    /// Byte size of one value.
    pub fn byte_size(self) -> usize {
        4 // both u32 and f32 are 4 bytes
    }
}

/// One named field inside a cell struct.
#[derive(Debug, Clone)]
pub struct FieldDef {
    /// Name as it will appear in the WGSL struct.
    pub name: String,
    pub ty: FieldType,
    /// Default value written as a WGSL literal (e.g. `"0u"`, `"0.0"`).
    pub default_wgsl: String,
}

impl FieldDef {
    pub fn new_u32(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ty: FieldType::U32,
            default_wgsl: "0u".into(),
        }
    }

    pub fn new_f32(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ty: FieldType::F32,
            default_wgsl: "0.0".into(),
        }
    }

    pub fn with_default(mut self, lit: impl Into<String>) -> Self {
        self.default_wgsl = lit.into();
        self
    }
}
