// ── sparse/active_set.rs ──────────────────────────────────────────────────────
//
// Sparse update system
// ─────────────────────
// Most cellular automata have large "dead" regions that don't change every
// step.  The sparse system skips those cells:
//
//   active_cells[]     – indices of cells to process this step
//   next_active_cells[]– cells that should be active next step
//   next_active_count  – atomic counter (single element, used by the GPU)
//
// After each compute dispatch:
//   1. Read back `next_active_count` (map buffer → copy to CPU).
//   2. The next dispatch only covers `next_active_count` threads.
//   3. Swap active ↔ next_active.
//
// The GPU shader handles writing to next_active_cells using the
// `activate_cell()` helper emitted by `ShaderBuilder`.
//
// All buffers are pre-allocated to `max_active_cells`.  If the active set
// overflows the buffer is silently truncated; choose `max_active_cells`
// conservatively (e.g. full cell_count).

#[allow(unused_imports)]
use bytemuck;
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub struct SparseActiveSet {
    /// Maximum number of simultaneously active cells.
    pub max_active: usize,
    /// Current active-cell count (CPU-side, updated after each readback).
    pub current_count: u32,

    /// GPU buffer: active cell indices for the *current* step.
    pub active_buf: wgpu::Buffer,
    /// GPU buffer: active cell indices for the *next* step.
    pub next_active_buf: wgpu::Buffer,
    /// GPU buffer: single `atomic<u32>` counting how many cells wrote to
    /// `next_active_buf` during the last dispatch.
    pub next_count_buf: wgpu::Buffer,
    /// Staging buffer for reading `next_count_buf` back to the CPU.
    pub readback_buf: wgpu::Buffer,
}

impl SparseActiveSet {
    /// Create the sparse buffers, seeded with the provided initial active indices.
    pub fn new(device: &Arc<wgpu::Device>, initial: &[u32], max_active: usize) -> Self {
        use wgpu::BufferUsages as Bu;

        // Pad the initial set to max_active with u32::MAX sentinels.
        let mut seed = vec![u32::MAX; max_active];
        let n = initial.len().min(max_active);
        seed[..n].copy_from_slice(&initial[..n]);

        let active_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sparse:active"),
            contents: bytemuck::cast_slice(&seed),
            usage: Bu::STORAGE | Bu::COPY_DST,
        });

        let next_active_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sparse:next_active"),
            size: (max_active * 4) as u64,
            usage: Bu::STORAGE | Bu::COPY_DST,
            mapped_at_creation: false,
        });

        // Atomic counter: 4 bytes for one u32.
        let next_count_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sparse:next_count"),
            contents: bytemuck::bytes_of(&0u32),
            usage: Bu::STORAGE | Bu::COPY_SRC | Bu::COPY_DST,
        });

        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sparse:readback"),
            size: 4,
            usage: Bu::MAP_READ | Bu::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            max_active,
            current_count: n as u32,
            active_buf,
            next_active_buf,
            next_count_buf,
            readback_buf,
        }
    }

    /// Encode commands to:
    ///   1. Copy `next_count_buf` → `readback_buf` (so the CPU can read it).
    ///   2. Clear `next_count_buf` back to zero for the next dispatch.
    pub fn encode_post_step(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue) {
        // Copy count to readback buffer
        encoder.copy_buffer_to_buffer(&self.next_count_buf, 0, &self.readback_buf, 0, 4);
        // Zero the counter
        queue.write_buffer(&self.next_count_buf, 0, bytemuck::bytes_of(&0u32));
    }

    /// Swap `active_buf` ↔ `next_active_buf` and update `current_count`.
    ///
    /// Call this **after** the GPU work for the step has completed and the
    /// readback future has resolved.  The engine must then rebuild its bind
    /// groups so they point at the newly-swapped buffers.
    pub fn swap_and_update(&mut self, new_count: u32) {
        std::mem::swap(&mut self.active_buf, &mut self.next_active_buf);
        self.current_count = new_count.min(self.max_active as u32);
    }

    /// Mark all cells as active (full-grid reset).
    pub fn activate_all(&mut self, queue: &wgpu::Queue, cell_count: usize) {
        let indices: Vec<u32> = (0..cell_count as u32).collect();
        queue.write_buffer(&self.active_buf, 0, bytemuck::cast_slice(&indices));
        self.current_count = cell_count as u32;
    }
}
