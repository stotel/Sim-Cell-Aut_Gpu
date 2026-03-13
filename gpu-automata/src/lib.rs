pub mod automata;
pub mod cell;
pub mod render;
pub mod rule_graph;
pub mod shader;
pub mod sparse;
pub mod topology;

pub use automata::engine::AutomataEngine;
pub use cell::{field::FieldType, schema::CellSchema};
pub use rule_graph::{graph::RuleGraph, node::NodeId};
pub use topology::{grid2d::SquareGrid2D, grid3d::CubicGrid3D, hex::HexGrid, Topology};
