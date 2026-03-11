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
}

impl SquareGrid2D {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            neighborhood: Neighborhood::Moore,
            wrapping: Wrapping::Torus,
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

    /// Convert (x, y) → linear cell index.
    #[inline]
    fn idx(&self, x: usize, y: usize) -> u32 {
        (y * self.width + x) as u32
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
                                // leave sentinel
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
}
