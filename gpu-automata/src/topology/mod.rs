// Neighbour table layout (flat GPU buffer):
//   neighbor_table[ cell_index * neighbor_count + k ] = index of the k-th neighbour,
//   or u32::MAX (0xFFFF_FFFF) when the slot is absent (boundary).

pub mod grid2d;
pub mod grid3d;
pub mod hex;

/// Core topology contract.
pub trait Topology: Send + Sync {
    /// Total number of cells in this topology.
    fn cell_count(&self) -> usize;

    /// Cells at boundaries may leave some slots as `u32::MAX`.
    fn neighbor_count(&self) -> usize;

    fn generate_neighbor_table(&self) -> Vec<u32>;

    /// Human-readable name used in debug labels.
    fn name(&self) -> &str;
}
