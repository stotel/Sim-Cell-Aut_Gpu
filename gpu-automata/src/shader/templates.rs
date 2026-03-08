// ── shader/templates.rs ───────────────────────────────────────────────────────
//
// Static WGSL fragments that are assembled by `ShaderBuilder`.
// Keeping them here makes it easy to audit shader code without reading Rust.

/// The workgroup size used for all compute shaders.
pub const WORKGROUP_SIZE: u32 = 256;

/// Render vertex-shader boilerplate (positions a unit quad per cell instance).
/// Requires the host to pass uniforms via `RenderUniforms`.
pub const RENDER_VERT_PREAMBLE: &str = r#"
struct RenderUniforms {
    grid_width:  u32,
    grid_height: u32,
    field_scale: f32,
    _pad:        u32,
}

@group(0) @binding(1) var<uniform> uniforms: RenderUniforms;

// Unit-quad vertex offsets (two triangles, CCW).
// IMPORTANT: must be var<private> — NOT const — because it is indexed
// with the runtime builtin vert_idx.  WGSL only permits compile-time
// constant indices into const-address-space arrays; var<private>
// supports arbitrary runtime indexing.
var<private> QUAD_VERTS: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-0.5, -0.5),
    vec2<f32>( 0.5, -0.5),
    vec2<f32>(-0.5,  0.5),
    vec2<f32>( 0.5, -0.5),
    vec2<f32>( 0.5,  0.5),
    vec2<f32>(-0.5,  0.5),
);

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0)       cell_value: f32,
}
"#;

/// Render fragment-shader boilerplate.
pub const RENDER_FRAG: &str = r#"
@fragment
fn fs_main(@location(0) cell_value: f32) -> @location(0) vec4<f32> {
    // Map [0, 1] cell value to a simple blue → white → yellow colour ramp.
    let t = clamp(cell_value, 0.0, 1.0);
    let cold = vec3<f32>(0.0, 0.2, 0.8);
    let hot  = vec3<f32>(1.0, 0.9, 0.1);
    return vec4<f32>(mix(cold, hot, t), 1.0);
}
"#;

/// Sparse-system helper: atomically append `cell_index` to `next_active_cells`.
pub const SPARSE_ACTIVATE_FN: &str = r#"
fn activate_cell(cell_index: u32) {
    let slot = atomicAdd(&next_active_count[0], 1u);
    if (slot < arrayLength(&next_active_cells)) {
        next_active_cells[slot] = cell_index;
    }
}
"#;
