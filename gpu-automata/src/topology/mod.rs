// Neighbour table layout (flat GPU buffer):
//   neighbor_table[ cell_index * neighbor_count + k ] = index of the k-th neighbour,
//   or u32::MAX (0xFFFF_FFFF) when the slot is absent (boundary).

pub mod grid2d;
pub mod grid3d;
pub mod hex;

pub type CellId = u32;
pub type ChunkId = u32;

/// Describes one GPU-to-GPU buffer copy for chunk boundary synchronisation.
#[derive(Clone, Debug)]
pub struct GpuBoundaryCopy {
    pub src_chunk: ChunkId,
    pub src_byte_offset: u64,
    pub dst_byte_offset: u64,
    pub byte_count: u64,
}

/// Boundary descriptor for one chunk.
#[derive(Clone, Debug)]
pub struct ChunkBoundaryDescriptor {
    pub cell_count: usize,
    pub copies: Vec<GpuBoundaryCopy>,
}

/// Core topology contract.
pub trait Topology: Send + Sync {
    /// Total number of cells in this topology.
    fn cell_count(&self) -> usize;

    /// Cells at boundaries may leave some slots as `u32::MAX`.
    fn neighbor_count(&self) -> usize;

    fn generate_neighbor_table(&self) -> Vec<u32>;

    /// Human-readable name used in debug labels.
    fn name(&self) -> &str;

    // ── Inline neighbour WGSL ─────────────────────────────────────────

    /// Return WGSL code defining `fn get_neighbor(cell_index: u32, slot: u32) -> u32`.
    /// When `Some`, the engine omits the neighbour-table buffer entirely.
    fn wgsl_neighbor_fn(&self) -> Option<String> {
        None
    }

    // ── GPU-resident chunking ─────────────────────────────────────────

    fn supports_gpu_chunks(&self) -> bool {
        false
    }
    fn chunk_count(&self) -> usize {
        1
    }
    fn chunk_own_count(&self, _chunk: ChunkId) -> usize {
        self.cell_count()
    }
    fn chunk_cells(&self, _chunk: ChunkId) -> Vec<CellId> {
        (0..self.cell_count() as CellId).collect()
    }
    fn chunk_boundary(
        &self,
        _chunk: ChunkId,
        _cell_byte_size: usize,
    ) -> Option<ChunkBoundaryDescriptor> {
        None
    }
    fn chunk_strip_height(&self, _chunk: ChunkId) -> u32 {
        0
    }
    fn chunk_strip_y0(&self, _chunk: ChunkId) -> u32 {
        0
    }

    /// Return WGSL for chunked-mode `fn get_neighbor`.
    fn wgsl_chunked_neighbor_fn(&self) -> Option<String> {
        None
    }

    /// Auto-configure chunk parameters to stay within `max_buffer_bytes`.
    fn auto_configure_chunks(&mut self, _max_buffer_bytes: u64, _cell_byte_size: usize) {}
}
