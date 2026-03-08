// ── rule_graph/node.rs ────────────────────────────────────────────────────────
//
// Every transformation step in a cellular-automaton rule is represented as a
// node in a directed acyclic graph (DAG).  The compiler in `compiler.rs` walks
// this DAG in topological order and emits one WGSL `let` binding per node.
//
// Node ID
// ───────
// `NodeId` is a newtype around a plain `usize` index into `RuleGraph::nodes`.
// Passing IDs by value makes graph construction ergonomic while remaining
// trivially copyable.
//
// Data type produced by each node
// ────────────────────────────────
// Every node carries an intrinsic `WgslType` so the compiler can cast when
// necessary (e.g. summing a `u32` field needs a cast to `f32` for the
// accumulator).

/// Opaque index into `RuleGraph::nodes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

/// Scalar types that nodes can produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WgslType {
    U32,
    F32,
    Bool,
}

impl WgslType {
    pub fn wgsl_keyword(self) -> &'static str {
        match self {
            WgslType::U32 => "u32",
            WgslType::F32 => "f32",
            WgslType::Bool => "bool",
        }
    }
}

/// Comparison operators available in `NodeKind::Compare`.
#[derive(Debug, Clone, Copy)]
pub enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CompareOp {
    pub fn wgsl_op(self) -> &'static str {
        match self {
            CompareOp::Eq => "==",
            CompareOp::Ne => "!=",
            CompareOp::Lt => "<",
            CompareOp::Le => "<=",
            CompareOp::Gt => ">",
            CompareOp::Ge => ">=",
        }
    }
}

/// The kind of computation a node performs.
///
/// Nodes are composed to express arbitrary cell-update logic.  The compiler
/// topologically sorts the graph and emits WGSL `let` bindings for each node
/// except `SetField` nodes, which emit direct assignments to `result_cell`.
#[derive(Debug, Clone)]
pub enum NodeKind {
    // ── Literals ────────────────────────────────────────────────────────────
    /// A constant 32-bit float.
    ConstantF32(f32),

    /// A constant 32-bit unsigned integer.
    ConstantU32(u32),

    // ── Arithmetic ──────────────────────────────────────────────────────────
    /// `a + b`  (both inputs must share a type)
    Add(NodeId, NodeId),

    /// `a - b`
    Sub(NodeId, NodeId),

    /// `a * b`
    Multiply(NodeId, NodeId),

    /// `a / b`
    Divide(NodeId, NodeId),

    // ── Logic / Comparison ──────────────────────────────────────────────────
    /// `a <op> b`  →  emits a `bool`
    Compare {
        lhs: NodeId,
        rhs: NodeId,
        op: CompareOp,
    },

    /// `if cond { if_true } else { if_false }`
    /// Produces the same type as `if_true` / `if_false` (must match).
    Select {
        cond: NodeId,
        if_true: NodeId,
        if_false: NodeId,
    },

    /// Logical AND of two bool nodes.
    And(NodeId, NodeId),

    /// Logical OR of two bool nodes.
    Or(NodeId, NodeId),

    /// Logical NOT of a bool node.
    Not(NodeId),

    // ── Cell accessors ──────────────────────────────────────────────────────
    /// Read a field from the current (self) cell.
    SelfField { field_name: String },

    /// Read a field from a specific neighbour slot (0 .. NEIGHBOR_COUNT-1).
    /// Emits `u32::MAX` guard so out-of-bounds slots are skipped.
    NeighborField { slot: usize, field_name: String },

    /// Sum a field across **all** valid neighbours.
    /// Always produces `f32` (casts u32 fields automatically).
    NeighborSum { field_name: String },

    /// Count neighbours where a `bool`-typed node (built per-neighbour) is true.
    /// Not yet implemented; reserved for future use.
    // NeighborCount { … }

    // ── Type casts ──────────────────────────────────────────────────────────
    /// Cast a value to `f32`.
    CastF32(NodeId),

    /// Cast a value to `u32` (truncation / bool→0/1).
    CastU32(NodeId),

    // ── Output ──────────────────────────────────────────────────────────────
    /// Write a computed value into a field of `result_cell`.
    /// Does **not** produce an output variable itself.
    SetField { field_name: String, value: NodeId },
}

/// A node in the rule DAG together with its output type.
#[derive(Debug, Clone)]
pub struct Node {
    pub kind: NodeKind,
    /// The WGSL type this node produces (None for `SetField`).
    pub output_type: Option<WgslType>,
}

impl Node {
    pub fn new(kind: NodeKind, output_type: Option<WgslType>) -> Self {
        Self { kind, output_type }
    }
}
