// gpu-automata: A high-performance GPU cellular automata engine built on WGPU.
//
// Architecture overview
// ─────────────────────
//   Topology   – describes the grid/graph structure and neighbour relationships.
//   CellSchema – describes the data fields stored per cell.
//   RuleGraph  – a data-driven computation graph that compiles to WGSL code.
//   ShaderBuilder – assembles the full compute shader from the above parts.
//   AutomataEngine – owns all GPU resources, runs simulation steps.
//   Renderer   – optional real-time visualisation.
//   SparseActiveSet – optional sparse-update bookkeeping.

pub mod automata;
pub mod cell;
pub mod render;
pub mod rule_graph;
pub mod shader;
pub mod sparse;
pub mod topology;

// ── convenience re-exports ────────────────────────────────────────────────────
pub use automata::engine::AutomataEngine;
pub use cell::{field::FieldType, schema::CellSchema};
pub use rule_graph::{graph::RuleGraph, node::NodeId};
pub use topology::{grid2d::SquareGrid2D, grid3d::CubicGrid3D, hex::HexGrid, Topology};
