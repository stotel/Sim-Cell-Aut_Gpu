// ── shader/templates.rs ───────────────────────────────────────────────────────

pub const WORKGROUP_SIZE: u32 = 256;

/// Render vertex-shader preamble.
/// Uses `CameraUniforms` for camera transform; grid dims are inside the struct.
pub const RENDER_VERT_PREAMBLE: &str = r#"
// Camera + grid uniforms (binding 1).
// cell_w / cell_h  – NDC size of one cell (encodes zoom + aspect ratio).
// cam_x  / cam_y   – camera centre in grid-cell units.
// grid_w / grid_h  – number of cells (needed to decode instance index).
struct CameraUniforms {
    cell_w : f32,
    cell_h : f32,
    cam_x  : f32,
    cam_y  : f32,
    grid_w : u32,
    grid_h : u32,
    _pad0  : u32,
    _pad1  : u32,
}

@group(0) @binding(1) var<uniform> camera: CameraUniforms;

// Unit-quad vertex offsets (two CCW triangles).
// Must be var<private> — WGSL const arrays only allow compile-time indices.
var<private> QUAD_VERTS: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-0.5, -0.5),
    vec2<f32>( 0.5, -0.5),
    vec2<f32>(-0.5,  0.5),
    vec2<f32>( 0.5, -0.5),
    vec2<f32>( 0.5,  0.5),
    vec2<f32>(-0.5,  0.5),
);

struct VertexOutput {
    @builtin(position) position  : vec4<f32>,
    @location(0)       cell_value: f32,
}
"#;

pub const RENDER_FRAG: &str = r#"
@fragment
fn fs_main(@location(0) cell_value: f32) -> @location(0) vec4<f32> {
    let t    = clamp(cell_value, 0.0, 1.0);
    let cold = vec3<f32>(0.0, 0.2, 0.8);
    let hot  = vec3<f32>(1.0, 0.9, 0.1);
    return vec4<f32>(mix(cold, hot, t), 1.0);
}
"#;

pub const SPARSE_ACTIVATE_FN: &str = r#"
fn activate_cell(cell_index: u32) {
    let slot = atomicAdd(&next_active_count[0], 1u);
    if (slot < arrayLength(&next_active_cells)) {
        next_active_cells[slot] = cell_index;
    }
}
"#;
