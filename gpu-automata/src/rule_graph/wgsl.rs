// ── rule_graph/wgsl.rs ────────────────────────────────────────────────────────
//
// Low-level WGSL text utilities used by the compiler and the shader builder.

/// Surround `body` with a WGSL block comment header/footer for readability.
pub fn section(title: &str, body: &str) -> String {
    format!(
        "// ── {title} {pad}\n{body}\n",
        title = title,
        pad = "-".repeat(60usize.saturating_sub(title.len() + 4)),
        body = body,
    )
}

/// Format a WGSL `const` declaration.
pub fn const_u32(name: &str, value: u32) -> String {
    format!("const {}: u32 = {}u;\n", name, value)
}

/// Format a WGSL `@group @binding` storage-buffer declaration.
pub fn storage_binding(
    group: u32,
    binding: u32,
    access: &str, // "read" | "read_write"
    name: &str,
    item_type: &str,
) -> String {
    format!(
        "@group({g}) @binding({b}) var<storage, {access}> {name}: array<{item}>;\n",
        g = group,
        b = binding,
        access = access,
        name = name,
        item = item_type,
    )
}

/// Format a WGSL `@group @binding` uniform-buffer declaration.
pub fn uniform_binding(group: u32, binding: u32, name: &str, struct_type: &str) -> String {
    format!(
        "@group({g}) @binding({b}) var<uniform> {name}: {ty};\n",
        g = group,
        b = binding,
        ty = struct_type,
        name = name,
    )
}
