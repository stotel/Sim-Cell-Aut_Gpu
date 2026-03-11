use super::field::{FieldDef, FieldType};

/// Defines the in-memory layout of a single cell, shared between CPU and GPU.
#[derive(Debug, Clone, Default)]
pub struct CellSchema {
    fields: Vec<FieldDef>,
}

impl CellSchema {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a field. Order determines the WGSL struct member order.
    pub fn add_field(mut self, field: FieldDef) -> Self {
        self.fields.push(field);
        self
    }

    /// Convenient builder for a `u32` field.
    pub fn field_u32(self, name: impl Into<String>) -> Self {
        self.add_field(FieldDef::new_u32(name))
    }

    /// Convenient builder for a `f32` field.
    pub fn field_f32(self, name: impl Into<String>) -> Self {
        self.add_field(FieldDef::new_f32(name))
    }

    /// Immutable view of the fields.
    pub fn fields(&self) -> &[FieldDef] {
        &self.fields
    }

    /// Byte stride of one cell (all fields are 4 bytes).
    pub fn cell_byte_size(&self) -> usize {
        self.fields.len() * 4
    }

    /// Total byte size for a buffer holding `count` cells.
    pub fn buffer_byte_size(&self, count: usize) -> u64 {
        (self.cell_byte_size() * count) as u64
    }

    /// Look up the type of a named field, if it exists.
    pub fn field_type(&self, name: &str) -> Option<FieldType> {
        self.fields.iter().find(|f| f.name == name).map(|f| f.ty)
    }

    /// Generate the WGSL `struct Cell { … }` declaration.
    pub fn generate_wgsl_struct(&self) -> String {
        let mut s = String::from("struct Cell {\n");
        for f in &self.fields {
            s.push_str(&format!("    {}: {},\n", f.name, f.ty.wgsl_type()));
        }
        s.push_str("}\n");
        s
    }

    /// Generate a WGSL zero-init expression, e.g. `Cell(0u, 0.0, 0.0)`.
    pub fn generate_wgsl_zero_init(&self) -> String {
        let inits: Vec<_> = self
            .fields
            .iter()
            .map(|f| f.default_wgsl.as_str())
            .collect();
        format!("Cell({})", inits.join(", "))
    }

    /// Produce a `Vec<u8>` with `count` zero-initialised cells for CPU upload.
    pub fn zero_buffer(&self, count: usize) -> Vec<u8> {
        vec![0u8; self.cell_byte_size() * count]
    }
}
