# gpu-automata

A high-performance GPU cellular automata engine written in Rust, built on [WGPU](https://wgpu.rs/).

## Features

| Feature | Description |
|---|---|
| **Any topology** | 2-D square, hex, 3-D cubic, or any `impl Topology` |
| **Multi-field cells** | `CellSchema` generates WGSL structs at runtime |
| **Rule graphs** | Data-driven DAG compiled to WGSL — no shader editing needed |
| **Double buffering** | Zero-copy ping-pong cell buffers on the GPU |
| **Sparse updates** | Optional active-cell list skips unchanged regions |
| **Instanced rendering** | One draw call for the entire grid |

---

## Quick start

```sh
cargo run --example game_of_life
```

Controls: `Space` pause/resume · `R` reset · `Escape` quit

---

## Architecture

```
gpu-automata/
├── src/
│   ├── topology/        # Grid geometry + neighbour tables
│   │   ├── grid2d.rs    # SquareGrid2D (Moore / VonNeumann, torus / clamp)
│   │   ├── hex.rs       # HexGrid (6-neighbour offset coords)
│   │   └── grid3d.rs    # CubicGrid3D (Moore-26 / VonNeumann-6)
│   ├── cell/
│   │   ├── field.rs     # FieldType (U32 | F32) + FieldDef
│   │   └── schema.rs    # CellSchema → WGSL struct + buffer sizing
│   ├── rule_graph/
│   │   ├── node.rs      # NodeKind enum (30+ node types)
│   │   ├── graph.rs     # RuleGraph builder with ergonomic helpers
│   │   ├── compiler.rs  # Topological emit → WGSL let-bindings
│   │   └── wgsl.rs      # Low-level WGSL text helpers
│   ├── shader/
│   │   ├── builder.rs   # ShaderBuilder assembles full compute/render shader
│   │   └── templates.rs # Static WGSL fragment constants
│   ├── automata/
│   │   ├── buffers.rs   # GpuBuffers (double-buffered cells + neighbour table)
│   │   ├── pipeline.rs  # ComputePipelineSet (pipeline + 2 bind groups)
│   │   └── engine.rs    # AutomataEngine public API
│   ├── sparse/
│   │   └── active_set.rs # SparseActiveSet GPU buffers + CPU tracking
│   └── render/
│       └── renderer.rs  # Instanced quad renderer (field → colour ramp)
└── examples/
    └── game_of_life.rs  # Conway GoL on a 128×128 torus
```

---

## Defining a simulation

### 1. Cell schema

```rust
let schema = CellSchema::new()
    .field_u32("alive")
    .field_f32("energy")
    .field_f32("signal");
```

Generates this WGSL automatically:
```wgsl
struct Cell {
    alive:  u32,
    energy: f32,
    signal: f32,
}
```

### 2. Topology

```rust
// 2-D 200×200 torus with 8 neighbours
let topology = Box::new(SquareGrid2D::new(200, 200));

// Hex grid
let topology = Box::new(HexGrid::new(100, 100));

// 3-D cubic
let topology = Box::new(CubicGrid3D::new(32, 32, 32));
```

### 3. Rule graph (Game of Life)

```rust
let mut g = RuleGraph::new();

let sum    = g.neighbor_sum("alive");          // f32: count live neighbours
let two    = g.const_f32(2.0);
let three  = g.const_f32(3.0);
let eq2    = g.compare(sum, two,   CompareOp::Eq);
let eq3    = g.compare(sum, three, CompareOp::Eq);
let survive = g.or(eq2, eq3);
let born    = eq3;
let self_v  = g.self_field("alive", WgslType::U32);
let zero    = g.const_u32(0);
let is_alive = g.compare(self_v, zero, CompareOp::Ne);
let next    = g.select(is_alive, survive, born);
let next_u  = g.cast_u32(next);
g.set_field("alive", next_u);
```

The compiler emits this as inline WGSL inside the compute shader — no
hand-written shader code required.

### 4. Engine

```rust
let engine = AutomataEngine::new(
    device, queue,
    topology, schema, &rule_graph,
    initial_cells,
    EngineConfig::default(),
);

engine.step();           // one tick
engine.step_n(100);      // 100 ticks
let raw = engine.current_cells(); // CPU readback (slow)
```

---

## Sparse mode

```rust
let config = EngineConfig {
    sparse:         true,
    initial_active: active_indices,
};
```

The GPU shader tracks which cells changed and propagates the active set
to neighbours automatically.  Only active cells are dispatched.

---

## Extending the engine

| Extension point | Where |
|---|---|
| New topology | Implement `topology::Topology` |
| New node kind | Add variant to `rule_graph::node::NodeKind` + arm in compiler |
| New colour ramp | Edit `shader/templates.rs` `RENDER_FRAG` |
| Multi-pass rules | Chain multiple `RuleGraph`s, one pipeline per pass |

---

## Dependencies

```toml
wgpu      = "0.20"
winit     = "0.29"
bytemuck  = "1.14"
pollster  = "0.3"
```
