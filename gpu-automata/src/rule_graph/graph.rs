use super::node::{CompareOp, Node, NodeId, NodeKind, WgslType};

#[derive(Debug, Default)]
pub struct RuleGraph {
    pub(crate) nodes: Vec<Node>,
}

impl RuleGraph {
    pub fn new() -> Self {
        Self::default()
    }

    fn push(&mut self, kind: NodeKind, ty: Option<WgslType>) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node::new(kind, ty));
        id
    }

    // ── Literals ──────────────────────────────────────────────────────────

    pub fn const_f32(&mut self, v: f32) -> NodeId {
        self.push(NodeKind::ConstantF32(v), Some(WgslType::F32))
    }

    pub fn const_u32(&mut self, v: u32) -> NodeId {
        self.push(NodeKind::ConstantU32(v), Some(WgslType::U32))
    }

    // ── Arithmetic ────────────────────────────────────────────────────────

    pub fn add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let ty = self.nodes[a.0].output_type;
        self.push(NodeKind::Add(a, b), ty)
    }

    pub fn sub(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let ty = self.nodes[a.0].output_type;
        self.push(NodeKind::Sub(a, b), ty)
    }

    pub fn mul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let ty = self.nodes[a.0].output_type;
        self.push(NodeKind::Multiply(a, b), ty)
    }

    pub fn div(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let ty = self.nodes[a.0].output_type;
        self.push(NodeKind::Divide(a, b), ty)
    }

    // ── Logic ─────────────────────────────────────────────────────────────

    pub fn compare(&mut self, lhs: NodeId, rhs: NodeId, op: CompareOp) -> NodeId {
        self.push(NodeKind::Compare { lhs, rhs, op }, Some(WgslType::Bool))
    }

    pub fn and(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.push(NodeKind::And(a, b), Some(WgslType::Bool))
    }

    pub fn or(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.push(NodeKind::Or(a, b), Some(WgslType::Bool))
    }

    pub fn not(&mut self, a: NodeId) -> NodeId {
        self.push(NodeKind::Not(a), Some(WgslType::Bool))
    }

    pub fn select(&mut self, cond: NodeId, if_true: NodeId, if_false: NodeId) -> NodeId {
        let ty = self.nodes[if_true.0].output_type;
        self.push(
            NodeKind::Select {
                cond,
                if_true,
                if_false,
            },
            ty,
        )
    }

    /// Read a field from the current cell.
    pub fn self_field(&mut self, field_name: impl Into<String>, ty: WgslType) -> NodeId {
        self.push(
            NodeKind::SelfField {
                field_name: field_name.into(),
            },
            Some(ty),
        )
    }

    /// Sum one field across all valid neighbours (result is always `f32`).
    pub fn neighbor_sum(&mut self, field_name: impl Into<String>) -> NodeId {
        self.push(
            NodeKind::NeighborSum {
                field_name: field_name.into(),
            },
            Some(WgslType::F32),
        )
    }

    /// Read one field from a fixed neighbour slot.
    pub fn neighbor_field(
        &mut self,
        slot: usize,
        field_name: impl Into<String>,
        ty: WgslType,
    ) -> NodeId {
        self.push(
            NodeKind::NeighborField {
                slot,
                field_name: field_name.into(),
            },
            Some(ty),
        )
    }

    // ── Type casts ────────────────────────────────────────────────────────

    pub fn cast_f32(&mut self, src: NodeId) -> NodeId {
        self.push(NodeKind::CastF32(src), Some(WgslType::F32))
    }

    pub fn cast_u32(&mut self, src: NodeId) -> NodeId {
        self.push(NodeKind::CastU32(src), Some(WgslType::U32))
    }

    /// Assign a computed value to a field in `result_cell`.
    pub fn set_field(&mut self, field_name: impl Into<String>, value: NodeId) -> NodeId {
        self.push(
            NodeKind::SetField {
                field_name: field_name.into(),
                value,
            },
            None,
        )
    }

    // ── Introspection ─────────────────────────────────────────────────────

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}
