// ── lif_parser.rs ─────────────────────────────────────────────────────────────
//
// Parses Game of Life pattern files into a flat list of alive-cell coordinates.
//
// Supported formats
// ─────────────────
//   Life 1.05  (.lif)   – #Life 1.05 header, #P anchors, . / * cells
//   Life 1.06  (.lif)   – #Life 1.06 header, coordinate pairs
//   Plaintext  (.cells) – ! comments, . dead, O alive
//   RLE        (.rle)   – run-length-encoded, b/o/$/ ! delimiters
//
// All formats produce `LifPattern` with coords relative to (0,0).
// The caller is responsible for centering / placing the pattern on the grid.

use anyhow::{bail, Context, Result};
use std::path::Path;

/// A parsed Game of Life pattern.
#[derive(Debug, Clone, Default)]
pub struct LifPattern {
    /// Alive cells in (col, row) order; coordinates may be negative.
    pub cells: Vec<(i32, i32)>,
    /// Optional human-readable name from the file metadata.
    pub name: Option<String>,
    /// Bounding-box width  (max_col - min_col + 1).
    pub width: i32,
    /// Bounding-box height (max_row - min_row + 1).
    pub height: i32,
}

impl LifPattern {
    /// Translate all coordinates so the top-left of the bounding box is (0,0).
    pub fn normalize(&mut self) {
        if self.cells.is_empty() {
            return;
        }
        let min_x = self.cells.iter().map(|c| c.0).min().unwrap();
        let min_y = self.cells.iter().map(|c| c.1).min().unwrap();
        for c in &mut self.cells {
            c.0 -= min_x;
            c.1 -= min_y;
        }
        self.recompute_bounds();
    }

    fn recompute_bounds(&mut self) {
        if self.cells.is_empty() {
            self.width = 0;
            self.height = 0;
            return;
        }
        let min_x = self.cells.iter().map(|c| c.0).min().unwrap();
        let max_x = self.cells.iter().map(|c| c.0).max().unwrap();
        let min_y = self.cells.iter().map(|c| c.1).min().unwrap();
        let max_y = self.cells.iter().map(|c| c.1).max().unwrap();
        self.width = max_x - min_x + 1;
        self.height = max_y - min_y + 1;
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Parse a pattern file.  The format is detected automatically.
pub fn parse_file(path: &Path) -> Result<LifPattern> {
    let content =
        std::fs::read_to_string(path).with_context(|| format!("cannot read {:?}", path))?;

    let pat = if content.starts_with("#Life 1.06") {
        parse_life106(&content)?
    } else if content.starts_with("#Life 1.05") || content.contains("#P ") {
        parse_life105(&content)?
    } else if content.trim_start().starts_with('#') || content.trim_start().starts_with('x') {
        // RLE files often start with # comments or x = …
        parse_rle(&content)?
    } else {
        // Default: try plaintext
        parse_plaintext(&content)?
    };

    Ok(pat)
}

/// Build a raw `cells_current` byte buffer for a `CellSchema { alive: u32 }`
/// grid of `grid_w × grid_h`, centring the pattern.
pub fn pattern_to_grid(pat: &LifPattern, grid_w: usize, grid_h: usize) -> Vec<u8> {
    let mut buf = vec![0u8; grid_w * grid_h * 4];

    if pat.cells.is_empty() {
        return buf;
    }

    // Normalize the pattern so top-left = (0,0).
    let min_x = pat.cells.iter().map(|c| c.0).min().unwrap_or(0);
    let min_y = pat.cells.iter().map(|c| c.1).min().unwrap_or(0);

    // Centering offsets.
    let off_x = (grid_w as i32 - pat.width) / 2 - min_x;
    let off_y = (grid_h as i32 - pat.height) / 2 - min_y;

    for &(cx, cy) in &pat.cells {
        let gx = (cx + off_x) as usize;
        let gy = (cy + off_y) as usize;
        if gx < grid_w && gy < grid_h {
            let idx = gy * grid_w + gx;
            buf[idx * 4..idx * 4 + 4].copy_from_slice(&1u32.to_le_bytes());
        }
    }
    buf
}

// ── Format parsers ────────────────────────────────────────────────────────────

/// Life 1.05 – supports multiple `#P` anchor blocks.
fn parse_life105(content: &str) -> Result<LifPattern> {
    let mut cells = Vec::new();
    let mut name = None;
    let mut anchor_x = 0i32;
    let mut anchor_y = 0i32;
    let mut row = 0i32;
    let mut in_block = false;

    for line in content.lines() {
        let line = line.trim_end();

        if line.starts_with("#Life") || line.is_empty() {
            continue;
        }
        if line.starts_with("#D") || line.starts_with("#C") {
            if name.is_none() {
                let s = line[2..].trim().to_string();
                if !s.is_empty() {
                    name = Some(s);
                }
            }
            continue;
        }
        if line.starts_with("#N") {
            name = Some(line[2..].trim().to_string());
            continue;
        }
        if line.starts_with("#P") {
            // New anchor block: #P <x> <y>
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 3 {
                bail!("Malformed #P line: {line}");
            }
            anchor_x = parts[1].parse().context("bad #P x")?;
            anchor_y = parts[2].parse().context("bad #P y")?;
            row = 0;
            in_block = true;
            continue;
        }
        if line.starts_with('#') {
            continue;
        } // unknown directive

        if in_block {
            for (col, ch) in line.chars().enumerate() {
                if ch == '*' || ch == 'o' || ch == 'O' {
                    cells.push((anchor_x + col as i32, anchor_y + row));
                }
            }
            row += 1;
        }
    }

    let mut pat = LifPattern {
        cells,
        name,
        width: 0,
        height: 0,
    };
    pat.recompute_bounds();
    Ok(pat)
}

/// Life 1.06 – one alive-cell coordinate pair per line.
fn parse_life106(content: &str) -> Result<LifPattern> {
    let mut cells = Vec::new();
    let mut name = None;

    for line in content.lines() {
        let line = line.trim();
        if line.starts_with("#Life") {
            continue;
        }
        if line.starts_with('#') {
            if name.is_none() {
                let s = line[1..].trim().to_string();
                if !s.is_empty() {
                    name = Some(s);
                }
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let x: i32 = parts
            .next()
            .context("missing x")?
            .parse()
            .context("bad x")?;
        let y: i32 = parts
            .next()
            .context("missing y")?
            .parse()
            .context("bad y")?;
        cells.push((x, y));
    }

    let mut pat = LifPattern {
        cells,
        name,
        width: 0,
        height: 0,
    };
    pat.recompute_bounds();
    Ok(pat)
}

/// Plaintext (.cells) – `!` comments, `.` dead, `O` or `*` alive.
fn parse_plaintext(content: &str) -> Result<LifPattern> {
    let mut cells = Vec::new();
    let mut name = None;
    let mut row = 0i32;

    for line in content.lines() {
        if line.starts_with('!') {
            if line.starts_with("!Name:") && name.is_none() {
                name = Some(line[6..].trim().to_string());
            }
            continue;
        }
        for (col, ch) in line.chars().enumerate() {
            if ch == 'O' || ch == '*' {
                cells.push((col as i32, row));
            }
        }
        row += 1;
    }

    let mut pat = LifPattern {
        cells,
        name,
        width: 0,
        height: 0,
    };
    pat.recompute_bounds();
    Ok(pat)
}

/// Run-length encoding (.rle).
///
/// Format:
/// ```text
/// # optional comment lines
/// x = W, y = H[, rule = B3/S23]
/// <rle data>!
/// ```
/// Symbols: `b` = dead, `o` = alive, `$` = end-of-row, `!` = end.
/// Numbers before a symbol are repeat counts.
fn parse_rle(content: &str) -> Result<LifPattern> {
    let mut cells = Vec::new();
    let mut name = None;
    let mut rle = String::new();

    for line in content.lines() {
        let t = line.trim();
        if t.starts_with('#') {
            if (t.starts_with("#N") || t.starts_with("#n")) && name.is_none() {
                let s = t[2..].trim().to_string();
                if !s.is_empty() {
                    name = Some(s);
                }
            }
            // #C / #c = comment, may contain the name
            if (t.starts_with("#C") || t.starts_with("#c")) && name.is_none() {
                let s = t[2..].trim().to_string();
                if !s.is_empty() {
                    name = Some(s);
                }
            }
            continue;
        }
        if t.starts_with("x ") || t.starts_with("x=") {
            // Header line – we don't need the dimensions, just skip.
            continue;
        }
        rle.push_str(t);
        if t.contains('!') {
            break;
        }
    }

    // Decode RLE string
    let mut x = 0i32;
    let mut y = 0i32;
    let mut count = String::new();

    for ch in rle.chars() {
        match ch {
            '0'..='9' => count.push(ch),
            'b' | 'B' => {
                x += count.parse::<i32>().unwrap_or(1);
                count.clear();
            }
            'o' | 'O' | 'A'..='Z' if ch != 'B' => {
                // 'o' and all uppercase letters (except B) = alive in standard RLE
                let n = count.parse::<i32>().unwrap_or(1);
                count.clear();
                for i in 0..n {
                    cells.push((x + i, y));
                }
                x += n;
            }
            '$' => {
                let n = count.parse::<i32>().unwrap_or(1);
                count.clear();
                y += n;
                x = 0;
            }
            '!' => break,
            _ => {
                count.clear();
            }
        }
    }

    let mut pat = LifPattern {
        cells,
        name,
        width: 0,
        height: 0,
    };
    pat.recompute_bounds();
    Ok(pat)
}
