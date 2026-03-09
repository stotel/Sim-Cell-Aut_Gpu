// Parses Game of Life pattern files into a flat list of alive-cell coordinates.
//
// Supported formats:
//   RLE  (.rle / .lif)  – run-length-encoded, b/o/$/ ! delimiters
//
// Header line format (first non-comment line):
//   x = 192, y = 69, rule = B2/S
//   x = 36,  y = 9,  rule = B3/S23
//
// `declared_w` / `declared_h` come from x= / y= in the header.
// `width`      / `height`     are the actual alive-cell bounding box.
// Use `effective_w/h()` (max of both) when sizing the grid.

use anyhow::{Context, Result};
use std::path::Path;

// ── Pattern ───────────────────────────────────────────────────────────────────

/// A parsed Game of Life / Life-like pattern.
#[derive(Debug, Clone, Default)]
pub struct LifPattern {
    /// Alive cells in (col, row); coordinates may be negative.
    pub cells: Vec<(i32, i32)>,
    /// Optional human-readable name from the file metadata.
    pub name: Option<String>,

    /// Width  declared in `x = …` header (0 if absent).
    pub declared_w: i32,
    /// Height declared in `y = …` header (0 if absent).
    pub declared_h: i32,

    /// Bounding-box width  of alive cells (max_col − min_col + 1).
    pub width: i32,
    /// Bounding-box height of alive cells (max_row − min_row + 1).
    pub height: i32,

    /// Rule string exactly as written, e.g. `"B3/S23"` or `"B2/S"`.
    /// Defaults to `"B3/S23"` when absent.
    pub rule: String,
}

impl LifPattern {
    /// Translate all coordinates so the bounding box starts at (0, 0).
    pub fn normalize(&mut self) {
        if self.cells.is_empty() { return; }
        let min_x = self.cells.iter().map(|c| c.0).min().unwrap();
        let min_y = self.cells.iter().map(|c| c.1).min().unwrap();
        for c in &mut self.cells { c.0 -= min_x; c.1 -= min_y; }
        self.recompute_bounds();
    }

    /// Grid width required to contain this pattern:
    /// max of `declared_w` (from file header) and `width` (alive-cell bbox).
    pub fn effective_w(&self) -> i32 { self.declared_w.max(self.width) }

    /// Grid height required to contain this pattern.
    pub fn effective_h(&self) -> i32 { self.declared_h.max(self.height) }

    fn recompute_bounds(&mut self) {
        if self.cells.is_empty() { self.width = 0; self.height = 0; return; }
        let min_x = self.cells.iter().map(|c| c.0).min().unwrap();
        let max_x = self.cells.iter().map(|c| c.0).max().unwrap();
        let min_y = self.cells.iter().map(|c| c.1).min().unwrap();
        let max_y = self.cells.iter().map(|c| c.1).max().unwrap();
        self.width  = max_x - min_x + 1;
        self.height = max_y - min_y + 1;
    }
}

// ── Rule parsing ──────────────────────────────────────────────────────────────

/// Birth / survival neighbor-count sets extracted from a rule string.
#[derive(Debug, Clone, Default)]
pub struct ParsedRule {
    /// Neighbor counts that cause a dead cell to be born.
    pub birth: Vec<u32>,
    /// Neighbor counts that allow a live cell to survive.
    pub survival: Vec<u32>,
    /// The raw rule string for display (e.g. `"B3/S23"`).
    pub raw: String,
}

/// Parse a `B…/S…` (or legacy `S…/B…`) rule string into birth/survival sets.
///
/// Examples
/// --------
/// `"B3/S23"` → birth=[3], survival=[2,3]
/// `"B2/S"`   → birth=[2], survival=[]
/// `"23/3"`   → survival=[2,3], birth=[3]  (old-style, S first)
///
/// Falls back to B3/S23 on any parse failure.
pub fn parse_rule_string(rule: &str) -> ParsedRule {
    let rule = rule.trim();
    let upper = rule.to_uppercase();

    // New-style: contains both 'B' and 'S' with a '/' separator
    if upper.contains('B') && upper.contains('S') && rule.contains('/') {
        let parts: Vec<&str> = rule.splitn(2, '/').collect();
        if parts.len() == 2 {
            let (b_part, s_part) = if parts[0].to_uppercase().contains('B') {
                (parts[0], parts[1])
            } else {
                (parts[1], parts[0])
            };
            return ParsedRule {
                birth:    digits_after_letter(b_part, 'B'),
                survival: digits_after_letter(s_part, 'S'),
                raw:      rule.to_string(),
            };
        }
    }

    // Old-style: pure digits/digits  (survival first, birth second)
    if rule.contains('/') {
        let parts: Vec<&str> = rule.splitn(2, '/').collect();
        if parts.len() == 2 {
            return ParsedRule {
                survival: digit_list(parts[0]),
                birth:    digit_list(parts[1]),
                raw:      rule.to_string(),
            };
        }
    }

    log::warn!("Cannot parse rule {:?}, defaulting to B3/S23", rule);
    ParsedRule { birth: vec![3], survival: vec![2, 3], raw: "B3/S23".into() }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Parse a pattern file.
pub fn parse_file(path: &Path) -> Result<LifPattern> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("cannot read {:?}", path))?;
    parse_rle(&content)
}

/// Build a flat GPU cell buffer (`u32` per cell, 1 = alive) for a grid of
/// `grid_w × grid_h`, centering the pattern.
///
/// `offset_x` / `offset_y` are extra pixel shifts applied *after* centering
/// (positive = right / down).  Pass `0, 0` for pure centering.
pub fn pattern_to_grid(
    pat:      &LifPattern,
    grid_w:   usize,
    grid_h:   usize,
    offset_x: i32,
    offset_y: i32,
) -> Vec<u8> {
    let mut buf = vec![0u8; grid_w * grid_h * 4];
    if pat.cells.is_empty() { return buf; }

    let min_x = pat.cells.iter().map(|c| c.0).min().unwrap_or(0);
    let min_y = pat.cells.iter().map(|c| c.1).min().unwrap_or(0);

    // Center using effective bounds (declared or computed, whichever is larger).
    let base_x = (grid_w as i32 - pat.effective_w()) / 2 - min_x + offset_x;
    let base_y = (grid_h as i32 - pat.effective_h()) / 2 - min_y + offset_y;

    for &(cx, cy) in &pat.cells {
        let gx = cx + base_x;
        let gy = cy + base_y;
        if gx >= 0 && gy >= 0 && (gx as usize) < grid_w && (gy as usize) < grid_h {
            let idx = gy as usize * grid_w + gx as usize;
            buf[idx * 4..idx * 4 + 4].copy_from_slice(&1u32.to_le_bytes());
        }
    }
    buf
}

// ── RLE parser ────────────────────────────────────────────────────────────────

fn parse_rle(content: &str) -> Result<LifPattern> {
    let mut cells      = Vec::new();
    let mut name       = None;
    let mut rule       = "B3/S23".to_string();
    let mut declared_w = 0i32;
    let mut declared_h = 0i32;
    let mut rle_data   = String::new();
    let mut past_header = false;

    for line in content.lines() {
        let t = line.trim();

        // ── Comment lines ─────────────────────────────────────────────────
        if t.starts_with('#') {
            if (t.starts_with("#N") || t.starts_with("#n")) && name.is_none() {
                let s = t[2..].trim().to_string();
                if !s.is_empty() { name = Some(s); }
            }
            if (t.starts_with("#C") || t.starts_with("#c")) && name.is_none() {
                let s = t[2..].trim().to_string();
                if !s.is_empty() { name = Some(s); }
            }
            continue;
        }

        // ── Header line: x = W, y = H[, rule = R] ────────────────────────
        // Recognized by containing an '=' and starting with 'x' (ignoring case).
        if !past_header && t.to_lowercase().trim_start().starts_with('x') && t.contains('=') {
            let (w, h, r) = parse_header_line(t);
            declared_w  = w;
            declared_h  = h;
            if let Some(rs) = r { rule = rs; }
            past_header = true;
            continue;
        }

        // ── RLE data ──────────────────────────────────────────────────────
        past_header = true;
        rle_data.push_str(t);
        if t.contains('!') { break; }
    }

    // Decode RLE
    let mut x     = 0i32;
    let mut y     = 0i32;
    let mut count = String::new();

    for ch in rle_data.chars() {
        match ch {
            '0'..='9' => count.push(ch),
            'b' | 'B' => {
                x += count.parse::<i32>().unwrap_or(1);
                count.clear();
            }
            '$' => {
                y += count.parse::<i32>().unwrap_or(1);
                count.clear();
                x = 0;
            }
            '!' => break,
            // 'o', 'O', and any uppercase letter except 'B' = alive
            c if c.is_ascii_alphabetic() && c.to_ascii_uppercase() != 'B' => {
                let n = count.parse::<i32>().unwrap_or(1);
                count.clear();
                for i in 0..n { cells.push((x + i, y)); }
                x += n;
            }
            _ => { count.clear(); }
        }
    }

    let mut pat = LifPattern {
        cells, name, rule, declared_w, declared_h,
        width: 0, height: 0,
    };
    pat.recompute_bounds();
    Ok(pat)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Parse `x = W, y = H[, rule = R]` → (W, H, Option<R>).
fn parse_header_line(line: &str) -> (i32, i32, Option<String>) {
    let mut w    = 0i32;
    let mut h    = 0i32;
    let mut rule = None;
    for part in line.split(',') {
        if let Some((k, v)) = part.trim().split_once('=') {
            match k.trim().to_lowercase().as_str() {
                "x"    => { w    = v.trim().parse().unwrap_or(0); }
                "y"    => { h    = v.trim().parse().unwrap_or(0); }
                "rule" => { rule = Some(v.trim().to_string()); }
                _      => {}
            }
        }
    }
    (w, h, rule)
}

fn digit_list(s: &str) -> Vec<u32> {
    let mut v: Vec<u32> = s.chars()
        .filter(|c| c.is_ascii_digit())
        .map(|c| c as u32 - '0' as u32)
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

fn digits_after_letter(s: &str, letter: char) -> Vec<u32> {
    let upper = s.to_uppercase();
    match upper.find(letter) {
        Some(pos) => digit_list(&s[pos + 1..]),
        None      => Vec::new(),
    }
}