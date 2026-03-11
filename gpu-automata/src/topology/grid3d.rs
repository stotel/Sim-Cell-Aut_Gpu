use super::Topology;
use crate::topology::grid2d::Wrapping;

#[derive(Debug, Clone, Copy)]
pub enum Neighborhood3D {
    /// 26 surrounding cells (full 3×3×3 minus self).
    Moore,
    /// 6 face-adjacent cells.
    VonNeumann,
}

impl Neighborhood3D {
    fn offsets(self) -> Vec<(i32, i32, i32)> {
        match self {
            Neighborhood3D::VonNeumann => vec![
                (-1, 0, 0),
                (1, 0, 0),
                (0, -1, 0),
                (0, 1, 0),
                (0, 0, -1),
                (0, 0, 1),
            ],
            Neighborhood3D::Moore => {
                let mut v = Vec::with_capacity(26);
                for dz in -1..=1i32 {
                    for dy in -1..=1i32 {
                        for dx in -1..=1i32 {
                            if dx == 0 && dy == 0 && dz == 0 {
                                continue;
                            }
                            v.push((dx, dy, dz));
                        }
                    }
                }
                v
            }
        }
    }
}

pub struct CubicGrid3D {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub neighborhood: Neighborhood3D,
    pub wrapping: Wrapping,
}

impl CubicGrid3D {
    pub fn new(width: usize, height: usize, depth: usize) -> Self {
        Self {
            width,
            height,
            depth,
            neighborhood: Neighborhood3D::Moore,
            wrapping: Wrapping::Torus,
        }
    }

    pub fn with_neighborhood(mut self, n: Neighborhood3D) -> Self {
        self.neighborhood = n;
        self
    }

    pub fn with_wrapping(mut self, w: Wrapping) -> Self {
        self.wrapping = w;
        self
    }

    #[inline]
    fn idx(&self, x: usize, y: usize, z: usize) -> u32 {
        (z * self.width * self.height + y * self.width + x) as u32
    }
}

impl Topology for CubicGrid3D {
    fn cell_count(&self) -> usize {
        self.width * self.height * self.depth
    }

    fn neighbor_count(&self) -> usize {
        self.neighborhood.offsets().len()
    }

    fn generate_neighbor_table(&self) -> Vec<u32> {
        let offsets = self.neighborhood.offsets();
        let nc = offsets.len();
        let total = self.cell_count();
        let mut tbl = vec![u32::MAX; total * nc];

        for z in 0..self.depth {
            for y in 0..self.height {
                for x in 0..self.width {
                    let ci = z * self.width * self.height + y * self.width + x;
                    for (k, &(dx, dy, dz)) in offsets.iter().enumerate() {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        let nz = z as i32 + dz;

                        let (nx, ny, nz) = match self.wrapping {
                            Wrapping::Torus => (
                                nx.rem_euclid(self.width as i32) as usize,
                                ny.rem_euclid(self.height as i32) as usize,
                                nz.rem_euclid(self.depth as i32) as usize,
                            ),
                            Wrapping::Clamp => {
                                if nx < 0
                                    || nx >= self.width as i32
                                    || ny < 0
                                    || ny >= self.height as i32
                                    || nz < 0
                                    || nz >= self.depth as i32
                                {
                                    tbl[ci * nc + k] = u32::MAX;
                                    continue;
                                }
                                (nx as usize, ny as usize, nz as usize)
                            }
                        };

                        tbl[ci * nc + k] = self.idx(nx, ny, nz);
                    }
                }
            }
        }
        tbl
    }

    fn name(&self) -> &str {
        "CubicGrid3D"
    }
}
