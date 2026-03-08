// ── topology/mod.rs ───────────────────────────────────────────────────────────
//
// The Topology trait is the sole abstraction over grid geometry.
//
// Any grid (2-D square, hex, 3-D cubic, arbitrary graph …) implements this
// trait and nothing else in the engine needs to know the concrete type.
//
// Neighbour table layout (stored in a flat GPU buffer)
// ────────────────────────────────────────────────────
//   neighbor_table[ cell_index * neighbor_count + k ]
//     = index of the k-th neighbour of cell `cell_index`
//       OR  u32::MAX  (0xFFFF_FFFF) when the slot is absent / boundary.
//
// The GPU shader uses NEIGHBOR_COUNT (a generated constant) to stride into the
// table without needing any additional length information.

pub mod grid2d;
pub mod grid3d;
pub mod hex;

/// Core topology contract.
pub trait Topology: Send + Sync {
    /// Total number of cells in this topology.
    fn cell_count(&self) -> usize;

    /// Fixed number of neighbour slots per cell.
    /// Cells at boundaries may leave some slots as `u32::MAX`.
    fn neighbor_count(&self) -> usize;

    /// Build the flat neighbour-index table (length = cell_count × neighbor_count).
    fn generate_neighbor_table(&self) -> Vec<u32>;

    /// Human-readable name for shader comments / debug labels.
    fn name(&self) -> &str;
}
