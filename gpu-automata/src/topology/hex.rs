// ── topology/hex.rs ───────────────────────────────────────────────────────────
//
// Hexagonal grid using "offset coordinates" (odd-row shift, flat-top columns).
//
// Each cell has exactly 6 neighbours.  Boundary cells use `u32::MAX` sentinels
// for absent slots when Wrapping::Clamp is chosen.
//
// Layout: row-major, cell_index = row * width + col.

use super::Topology;
use crate::topology::grid2d::Wrapping;

/// Axial offset for each of the 6 hex directions in offset coordinates.
/// These differ for even / odd rows (the "pointy-top" odd-row convention).
const EVEN_ROW_OFFSETS: [(i32, i32); 6] = [(1, 0), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)];
const ODD_ROW_OFFSETS: [(i32, i32); 6] = [(1, 0), (1, -1), (0, -1), (-1, 0), (0, 1), (1, 1)];

pub struct HexGrid {
    pub width: usize,
    pub height: usize,
    pub wrapping: Wrapping,
}

impl HexGrid {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            wrapping: Wrapping::Torus,
        }
    }

    pub fn with_wrapping(mut self, w: Wrapping) -> Self {
        self.wrapping = w;
        self
    }
}

impl Topology for HexGrid {
    fn cell_count(&self) -> usize {
        self.width * self.height
    }
    fn neighbor_count(&self) -> usize {
        6
    }

    fn generate_neighbor_table(&self) -> Vec<u32> {
        let nc = 6usize;
        let total = self.cell_count();
        let mut tbl = vec![u32::MAX; total * nc];

        for row in 0..self.height {
            for col in 0..self.width {
                let ci = row * self.width + col;
                let offsets = if row % 2 == 0 {
                    &EVEN_ROW_OFFSETS
                } else {
                    &ODD_ROW_OFFSETS
                };

                for (k, &(dc, dr)) in offsets.iter().enumerate() {
                    let nc_col = col as i32 + dc;
                    let nr_row = row as i32 + dr;

                    let (nc_col, nr_row) = match self.wrapping {
                        Wrapping::Torus => (
                            nc_col.rem_euclid(self.width as i32) as usize,
                            nr_row.rem_euclid(self.height as i32) as usize,
                        ),
                        Wrapping::Clamp => {
                            if nc_col < 0
                                || nc_col >= self.width as i32
                                || nr_row < 0
                                || nr_row >= self.height as i32
                            {
                                tbl[ci * nc + k] = u32::MAX;
                                continue;
                            }
                            (nc_col as usize, nr_row as usize)
                        }
                    };

                    tbl[ci * nc + k] = (nr_row * self.width + nc_col) as u32;
                }
            }
        }
        tbl
    }

    fn name(&self) -> &str {
        "HexGrid"
    }
}
