// Compiles a `RuleGraph` into a WGSL code
//
// Code generation
// 1. Walk nodes in their stored order
// 2. For every non-terminal node, emit:
//       let var_<id>: <WgslType> = <expression>;
// 3. For every `SetField` node, emit:
//       result_cell.<field> = var_<value_id>;
//
// The emitted snippet assumes these identifiers exist in the surrounding
// shader:
// `self_cell`       – current cell (read from `cells_current`)
// `result_cell`     – mutable copy that will be written to `cells_next`
// `cell_index`      – `u32` global invocation index
// `read_cell(idx)`  – reads a cell (dispatches to own cells or boundary)
// `get_neighbor(cell_index, slot)` – returns neighbour cell index
// `NEIGHBOR_COUNT`  – compile-time constant for neighbour count

use super::graph::RuleGraph;
use super::node::{NodeId, NodeKind, WgslType};

///Output of the rule graph compiler.
pub struct CompiledRule {
    ///WGSL statements
    pub wgsl_body: String,
}

/// Format an f32 as a WGSL float literal.
fn format_f32(v: f32) -> String {
    if v.is_nan() {
        // WGSL has no NaN literal; use divide-by-zero
        "( 0.0f / 0.0f)".to_string()
    } else if v.is_infinite() {
        if v > 0.0 {
            "( 1.0f / 0.0f)".to_string()
        } else {
            "(-1.0f / 0.0f)".to_string()
        }
    } else {
        let s = format!("{:?}", v);
        format!("{}f", s)
    }
}

pub struct RuleCompiler<'a> {
    graph: &'a RuleGraph,
}

impl<'a> RuleCompiler<'a> {
    pub fn new(graph: &'a RuleGraph) -> Self {
        Self { graph }
    }

    ///Compile the graph to a WGSL snippet.
    pub fn compile(&self) -> CompiledRule {
        let mut out = String::new();
        out.push_str("    // ── Begin generated rule graph ────────────────────\n");

        for (i, node) in self.graph.nodes.iter().enumerate() {
            let id = NodeId(i);
            let var = Self::var(id);

            let line = match &node.kind {
                //Literals
                NodeKind::ConstantF32(v) => {
                    let lit = format_f32(*v);
                    format!("    let {var}: f32 = {lit};\n", var = var, lit = lit)
                }
                NodeKind::ConstantU32(v) => {
                    format!("    let {var}: u32 = {v}u;\n", var = var, v = v)
                }

                //Arithmetic
                NodeKind::Add(a, b) => {
                    let ty = self.ty(node.output_type);
                    format!(
                        "    let {var}: {ty} = {a} + {b};\n",
                        var = var,
                        ty = ty,
                        a = Self::var(*a),
                        b = Self::var(*b)
                    )
                }
                NodeKind::Sub(a, b) => {
                    let ty = self.ty(node.output_type);
                    format!(
                        "    let {var}: {ty} = {a} - {b};\n",
                        var = var,
                        ty = ty,
                        a = Self::var(*a),
                        b = Self::var(*b)
                    )
                }
                NodeKind::Multiply(a, b) => {
                    let ty = self.ty(node.output_type);
                    format!(
                        "    let {var}: {ty} = {a} * {b};\n",
                        var = var,
                        ty = ty,
                        a = Self::var(*a),
                        b = Self::var(*b)
                    )
                }
                NodeKind::Divide(a, b) => {
                    let ty = self.ty(node.output_type);
                    format!(
                        "    let {var}: {ty} = {a} / {b};\n",
                        var = var,
                        ty = ty,
                        a = Self::var(*a),
                        b = Self::var(*b)
                    )
                }

                //Logic
                NodeKind::Compare { lhs, rhs, op } => {
                    format!(
                        "    let {var}: bool = {lhs} {op} {rhs};\n",
                        var = var,
                        lhs = Self::var(*lhs),
                        op = op.wgsl_op(),
                        rhs = Self::var(*rhs)
                    )
                }
                NodeKind::And(a, b) => {
                    format!(
                        "    let {var}: bool = {a} && {b};\n",
                        var = var,
                        a = Self::var(*a),
                        b = Self::var(*b)
                    )
                }
                NodeKind::Or(a, b) => {
                    format!(
                        "    let {var}: bool = {a} || {b};\n",
                        var = var,
                        a = Self::var(*a),
                        b = Self::var(*b)
                    )
                }
                NodeKind::Not(a) => {
                    format!(
                        "    let {var}: bool = !{a};\n",
                        var = var,
                        a = Self::var(*a)
                    )
                }
                NodeKind::Select {
                    cond,
                    if_true,
                    if_false,
                } => {
                    let ty = self.ty(node.output_type);
                    //WGSL `select(false_val, true_val, cond)`
                    format!(
                        "    let {var}: {ty} = select({f}, {t}, {c});\n",
                        var = var,
                        ty = ty,
                        c = Self::var(*cond),
                        t = Self::var(*if_true),
                        f = Self::var(*if_false)
                    )
                }

                //Cell accessors
                NodeKind::SelfField { field_name } => {
                    let ty = self.ty(node.output_type);
                    format!(
                        "    let {var}: {ty} = self_cell.{field};\n",
                        var = var,
                        ty = ty,
                        field = field_name
                    )
                }
                NodeKind::NeighborField { slot, field_name } => {
                    let ty = self.ty(node.output_type);
                    //Guard against absent neighbours
                    let mut s = String::new();
                    s.push_str(&format!(
                        "    var {var}: {ty} = {zero};\n",
                        var = var,
                        ty = ty,
                        zero = match node.output_type {
                            Some(WgslType::F32) => "0.0",
                            _ => "0u",
                        }
                    ));
                    s.push_str(&format!(
                        "    {{\n        let _nb_idx_{i}: u32 = get_neighbor(cell_index, {slot}u);\n        if (_nb_idx_{i} != 0xFFFFFFFFu) {{ {var} = read_cell(_nb_idx_{i}).{field}; }}\n    }}\n",
                        i    = i,
                        slot = slot,
                        var  = var,
                        field = field_name,
                    ));
                    s
                }

                NodeKind::NeighborSum { field_name } => {
                    //Emit a WGSL loop that sums the field over all valid neighbours.
                    let acc = format!("_nb_sum_{i}", i = i);
                    let mut s = String::new();
                    s.push_str(&format!(
                        "    var {acc}: f32 = 0.0;\n
                             for (var _ni_{i}: u32 = 0u; _ni_{i} < NEIGHBOR_COUNT; _ni_{i}++) {{\n
                                let _nidx_{i}: u32 = get_neighbor(cell_index, _ni_{i});\n 
                                if (_nidx_{i} != 0xFFFFFFFFu) {{\n
                                    {acc} = {acc} + f32(read_cell(_nidx_{i}).{field});\n
                                }}\n
                             }}\n    
                             let {var}: f32 = {acc};\n",
                        acc   = acc,
                        i     = i,
                        field = field_name,
                        var   = var,
                    ));
                    s
                }

                //Type casts
                NodeKind::CastF32(src) => {
                    format!(
                        "    let {var}: f32 = f32({src});\n",
                        var = var,
                        src = Self::var(*src)
                    )
                }
                NodeKind::CastU32(src) => {
                    format!(
                        "    let {var}: u32 = u32({src});\n",
                        var = var,
                        src = Self::var(*src)
                    )
                }

                //Output
                NodeKind::SetField { field_name, value } => {
                    format!(
                        "    result_cell.{field} = {val};\n",
                        field = field_name,
                        val = Self::var(*value)
                    )
                }
            };

            out.push_str(&line);
        }

        out.push_str("    // ── End generated rule graph ──────────────────────\n");
        CompiledRule { wgsl_body: out }
    }

    //Helpers

    fn var(id: NodeId) -> String {
        format!("_r{}", id.0)
    }

    fn ty(&self, t: Option<WgslType>) -> &'static str {
        match t {
            Some(WgslType::F32) => "f32",
            Some(WgslType::U32) => "u32",
            Some(WgslType::Bool) => "bool",
            None => "/*void*/",
        }
    }
}
