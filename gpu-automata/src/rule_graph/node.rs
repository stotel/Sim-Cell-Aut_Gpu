#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

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

///The kind of computation a node performs.
///Nodes are composed to express arbitrary cell-update logic.  The compiler
///topologically sorts the graph and emits WGSL `let` bindings for each node
///except `SetField` nodes, which emit direct assignments to `result_cell`.
#[derive(Debug, Clone)]
pub enum NodeKind {
    /// A constant f32
    ConstantF32(f32),

    /// A constant u32
    ConstantU32(u32),

    /// `a + b`  
    Add(NodeId, NodeId),

    /// `a - b`
    Sub(NodeId, NodeId),

    /// `a * b`
    Multiply(NodeId, NodeId),

    /// `a / b`
    Divide(NodeId, NodeId),

    /// `a <op> b` gives `bool`
    Compare {
        lhs: NodeId,
        rhs: NodeId,
        op: CompareOp,
    },

    ///`if cond { if_true } else { if_false }`
    ///Produces the same type as `if_true` / `if_false` (must match).
    Select {
        cond: NodeId,
        if_true: NodeId,
        if_false: NodeId,
    },

    ///Logical AND of two bool nodes.
    And(NodeId, NodeId),

    ///Logical OR of two bool nodes.
    Or(NodeId, NodeId),

    ///Logical NOT of a bool node.
    Not(NodeId),

    ///Read a field from the current (self) cell.
    SelfField { field_name: String },

    ///Read a field from a specific neighbour slot (0 .. NEIGHBOR_COUNT-1).
    ///Emits `u32::MAX` guard so out-of-bounds slots are skipped.
    NeighborField { slot: usize, field_name: String },

    ///Sum a field across **all** valid neighbours.
    ///Always produces `f32` (casts u32 fields automatically).
    NeighborSum { field_name: String },

    ///Cast a value to `f32`.
    CastF32(NodeId),

    ///Cast a value to `u32` (truncation / bool→0/1).
    CastU32(NodeId),

    ///Write a computed value into a field of `result_cell`.
    ///Does **not** produce an output variable itself.
    SetField { field_name: String, value: NodeId },
}

///A node in the rule DAG together with its output type.
#[derive(Debug, Clone)]
pub struct Node {
    pub kind: NodeKind,
    ///The WGSL type this node produces (`None` for `SetField`).
    pub output_type: Option<WgslType>,
}

impl Node {
    pub fn new(kind: NodeKind, output_type: Option<WgslType>) -> Self {
        Self { kind, output_type }
    }
}
