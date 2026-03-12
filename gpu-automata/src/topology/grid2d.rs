use super::Topology;

/// How the grid handles its boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wrapping {
    /// Edges connect to the opposite side (torus topology).
    Torus,
    /// Out-of-bound neighbours are marked absent (`u32::MAX`).
    Clamp,
}

/// Which cells are considered neighbours.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Neighborhood {
    /// 8 surrounding cells (includes diagonals).
    Moore,
    /// 4 cardinal cells (N, E, S, W).
    VonNeumann,
}

impl Neighborhood {
    fn offsets(self) -> &'static [(i32, i32)] {
        match self {
            Neighborhood::Moore => &[
                (-1, -1),
                (0, -1),
                (1, -1),
                (-1, 0),
                (1, 0),
                (-1, 1),
                (0, 1),
                (1, 1),
            ],
            Neighborhood::VonNeumann => &[(0, -1), (-1, 0), (1, 0), (0, 1)],
        }
    }
}

/// 2-D rectangular grid topology.
pub struct SquareGrid2D {
    pub width: usize,
    pub height: usize,
    pub neighborhood: Neighborhood,
    pub wrapping: Wrapping,
    /// When `Some(r)`, the grid is split into horizontal strips of `r` rows
    /// (the last strip may be shorter).  Each strip becomes one GPU chunk.
    pub chunk_rows: Option<usize>,
}

impl SquareGrid2D {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            neighborhood: Neighborhood::Moore,
            wrapping: Wrapping::Torus,
            chunk_rows: None,
        }
    }

    pub fn with_neighborhood(mut self, n: Neighborhood) -> Self {
        self.neighborhood = n;
        self
    }

    pub fn with_wrapping(mut self, w: Wrapping) -> Self {
        self.wrapping = w;
        self
    }

    pub fn with_chunk_rows(mut self, rows: usize) -> Self {
        self.chunk_rows = Some(rows);
        self
    }

    /// Number of horizontal strips.
    pub fn strip_count(&self) -> usize {
        match self.chunk_rows {
            Some(cr) => (self.height + cr - 1) / cr,
            None => 1,
        }
    }

    /// Number of rows in strip `c`.
    fn strip_rows(&self, c: usize) -> usize {
        let cr = self.chunk_rows.unwrap_or(self.height);
        let y0 = c * cr;
        cr.min(self.height - y0)
    }

    /// First global row of strip `c`.
    fn strip_y0_val(&self, c: usize) -> usize {
        let cr = self.chunk_rows.unwrap_or(self.height);
        c * cr
    }

    /// Convert (x, y) → linear cell index.
    #[inline]
    fn idx(&self, x: usize, y: usize) -> u32 {
        (y * self.width + x) as u32
    }

    /// Generate the WGSL `switch` cases for the current neighbourhood.
    fn wgsl_switch_body(&self) -> String {
        let mut s = String::new();
        for (k, &(dx, dy)) in self.neighborhood.offsets().iter().enumerate() {
            s.push_str(&format!(
                "        case {k}u: {{ dx = {dx}; dy = {dy}; }}\n",
                k = k, dx = dx, dy = dy,
            ));
        }
        s.push_str("        default: { return 0xFFFFFFFFu; }\n");
        s
    }
}

impl Topology for SquareGrid2D {
    fn cell_count(&self) -> usize {
        self.width * self.height
    }

    fn neighbor_count(&self) -> usize {
        self.neighborhood.offsets().len()
    }

    fn generate_neighbor_table(&self) -> Vec<u32> {
        let nc = self.neighbor_count();
        let total = self.cell_count();
        let mut tbl = vec![u32::MAX; total * nc];

        for y in 0..self.height {
            for x in 0..self.width {
                let ci = y * self.width + x;
                for (k, &(dx, dy)) in self.neighborhood.offsets().iter().enumerate() {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;

                    let (nx, ny) = match self.wrapping {
                        Wrapping::Torus => (
                            nx.rem_euclid(self.width as i32) as usize,
                            ny.rem_euclid(self.height as i32) as usize,
                        ),
                        Wrapping::Clamp => {
                            if nx < 0
                                || nx >= self.width as i32
                                || ny < 0
                                || ny >= self.height as i32
                            {
                                tbl[ci * nc + k] = u32::MAX;
                                continue;
                            }
                            (nx as usize, ny as usize)
                        }
                    };

                    tbl[ci * nc + k] = self.idx(nx, ny);
                }
            }
        }
        tbl
    }

    fn name(&self) -> &str {
        "SquareGrid2D"
    }

    // ── Inline neighbour WGSL ────────────────────────────────────────────

    fn wgsl_neighbor_fn(&self) -> Option<String> {
        let w = self.width;
        let h = self.height;
        let mut s = String::new();
        s.push_str("fn get_neighbor(cell_index: u32, slot: u32) -> u32 {\n");
        s.push_str(&format!("    let W: u32 = {}u;\n", w));
        s.push_str(&format!("    let H: u32 = {}u;\n", h));
        s.push_str("    let x: u32 = cell_index % W;\n");
        s.push_str("    let y: u32 = cell_index / W;\n");
        s.push_str("    var dx: i32; var dy: i32;\n");
        s.push_str("    switch (slot) {\n");
        s.push_str(&self.wgsl_switch_body());
        s.push_str("    }\n");
        s.push_str("    let nx: i32 = i32(x) + dx;\n");
        s.push_str("    let ny: i32 = i32(y) + dy;\n");
        match self.wrapping {
            Wrapping::Torus => {
                s.push_str("    let wx: u32 = u32((nx % i32(W) + i32(W)) % i32(W));\n");
                s.push_str("    let wy: u32 = u32((ny % i32(H) + i32(H)) % i32(H));\n");
                s.push_str("    return wy * W + wx;\n");
            }
            Wrapping::Clamp => {
                s.push_str("    if (nx < 0 || nx >= i32(W) || ny < 0 || ny >= i32(H)) { return 0xFFFFFFFFu; }\n");
                s.push_str("    return u32(ny) * W + u32(nx);\n");
            }
        }
        s.push_str("}\n");
        Some(s)
    }

    // ── GPU-resident chunking ────────────────────────────────────────────

    fn supports_gpu_chunks(&self) -> bool {
        self.chunk_rows.is_some()
    }

    fn chunk_count(&self) -> usize {
        self.strip_count()
    }

    fn chunk_own_count(&self, chunk: super::ChunkId) -> usize {
        self.strip_rows(chunk as usize) * self.width
    }

    fn chunk_cells(&self, chunk: super::ChunkId) -> Vec<super::CellId> {
        let c = chunk as usize;
        let y0 = self.strip_y0_val(c);
        let rows = self.strip_rows(c);
        let start = (y0 * self.width) as super::CellId;
        let count = (rows * self.width) as super::CellId;
        (start..start + count).collect()
    }

    fn chunk_boundary(
        &self,
        chunk: super::ChunkId,
        cell_byte_size: usize,
    ) -> Option<super::ChunkBoundaryDescriptor> {
        let _cr = self.chunk_rows?;
        let w = self.width;
        let n = self.strip_count();
        let c = chunk as usize;
        let row_bytes = (w * cell_byte_size) as u64;

        let mut copies = Vec::new();

        // Top boundary: last row of the chunk above
        match self.wrapping {
            Wrapping::Torus => {
                let src_c = if c == 0 { n - 1 } else { c - 1 };
                let src_h = self.strip_rows(src_c);
                copies.push(super::GpuBoundaryCopy {
                    src_chunk: src_c as super::ChunkId,
                    src_byte_offset: ((src_h - 1) * w * cell_byte_size) as u64,
                    dst_byte_offset: 0,
                    byte_count: row_bytes,
                });
            }
            Wrapping::Clamp => {
                if c > 0 {
                    let src_c = c - 1;
                    let src_h = self.strip_rows(src_c);
                    copies.push(super::GpuBoundaryCopy {
                        src_chunk: src_c as super::ChunkId,
                        src_byte_offset: ((src_h - 1) * w * cell_byte_size) as u64,
                        dst_byte_offset: 0,
                        byte_count: row_bytes,
                    });
                }
            }
        }

        // Bottom boundary: first row of the chunk below
        match self.wrapping {
            Wrapping::Torus => {
                let src_c = if c == n - 1 { 0 } else { c + 1 };
                copies.push(super::GpuBoundaryCopy {
                    src_chunk: src_c as super::ChunkId,
                    src_byte_offset: 0,
                    dst_byte_offset: row_bytes,
                    byte_count: row_bytes,
                });
            }
            Wrapping::Clamp => {
                if c < n - 1 {
                    copies.push(super::GpuBoundaryCopy {
                        src_chunk: (c + 1) as super::ChunkId,
                        src_byte_offset: 0,
                        dst_byte_offset: row_bytes,
                        byte_count: row_bytes,
                    });
                }
            }
        }

        Some(super::ChunkBoundaryDescriptor {
            cell_count: 2 * w,
            copies,
        })
    }

    fn chunk_strip_height(&self, chunk: super::ChunkId) -> u32 {
        self.strip_rows(chunk as usize) as u32
    }

    fn chunk_strip_y0(&self, chunk: super::ChunkId) -> u32 {
        self.strip_y0_val(chunk as usize) as u32
    }

    fn wgsl_chunked_neighbor_fn(&self) -> Option<String> {
        if self.chunk_rows.is_none() {
            return None;
        }
        let w = self.width;
        let h = self.height;
        let mut s = String::new();
        s.push_str("fn get_neighbor(local_idx: u32, slot: u32) -> u32 {\n");
        s.push_str(&format!("    let W: u32 = {}u;\n", w));
        s.push_str(&format!("    let H: u32 = {}u;\n", h));
        s.push_str("    let strip_h: u32 = chunk_params.strip_h;\n");
        s.push_str("    let strip_y0: u32 = chunk_params.strip_y0;\n");
        s.push_str("    let x: u32 = local_idx % W;\n");
        s.push_str("    let y: u32 = local_idx / W;\n");
        s.push_str("    var dx: i32; var dy: i32;\n");
        s.push_str("    switch (slot) {\n");
        s.push_str(&self.wgsl_switch_body());
        s.push_str("    }\n");
        s.push_str("    let nx: i32 = i32(x) + dx;\n");
        s.push_str("    let ny: i32 = i32(y) + dy;\n");
        // X wrapping
        match self.wrapping {
            Wrapping::Torus => {
                s.push_str("    let wx: u32 = u32((nx % i32(W) + i32(W)) % i32(W));\n");
            }
            Wrapping::Clamp => {
                s.push_str("    if (nx < 0 || nx >= i32(W)) { return 0xFFFFFFFFu; }\n");
                s.push_str("    let wx: u32 = u32(nx);\n");
            }
        }
        // Y: check global bounds, then check boundary
        s.push_str("    let global_y: i32 = i32(strip_y0) + ny;\n");
        match self.wrapping {
            Wrapping::Clamp => {
                s.push_str("    if (global_y < 0 || global_y >= i32(H)) { return 0xFFFFFFFFu; }\n");
            }
            Wrapping::Torus => {} // boundary copies handle torus wrap
        }
        // If y is above or below the strip, index into the boundary buffer
        // Boundary layout: [top_row (W cells) | bottom_row (W cells)]
        s.push_str("    if (ny < 0) { return 0x80000000u | wx; }\n");
        s.push_str("    if (ny >= i32(strip_h)) { return 0x80000000u | (W + wx); }\n");
        s.push_str("    return u32(ny) * W + wx;\n");
        s.push_str("}\n");
        Some(s)
    }

    fn auto_configure_chunks(&mut self, max_buffer_bytes: u64, cell_byte_size: usize) {
        let total_bytes = (self.width * self.height * cell_byte_size) as u64;
        if total_bytes <= max_buffer_bytes {
            return;
        }
        let row_bytes = (self.width * cell_byte_size) as u64;
        // Use 90% of max to leave room for boundary buffers / uniforms
        let max_rows = ((max_buffer_bytes * 9 / 10) / row_bytes) as usize;
        self.chunk_rows = Some(max_rows.max(1));
    }
}
