use routingkit_cch::{CCH, CCHMetric as RoutingMetric, CCHQuery, compute_order_inertial};

pub const CCH_INFINITY: u32 = i32::MAX as u32;

/// Representation-neutral CCH topology.
///
/// The graph-representation layer owns the meaning of nodes, arcs, endpoints,
/// and learned coordinates. This type only owns the routing topology and its
/// deterministic elimination order.
pub(crate) struct CchTopology {
    cch: CCH,
    node_count: usize,
    arc_count: usize,
    order: Vec<u32>,
}

/// One full customization of a [`CchTopology`].
pub(crate) struct CchMetric<'a> {
    inner: RoutingMetric<'a>,
    node_count: usize,
}

/// Reusable query state for one customized metric.
pub(crate) struct CchReusableQuery<'a> {
    inner: CCHQuery<'a>,
    node_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CchPath {
    pub distance: u32,
    pub node_path: Vec<usize>,
    pub arc_path: Vec<usize>,
}

impl CchTopology {
    pub(crate) fn build(
        node_count: usize,
        tail: &[u32],
        head: &[u32],
        x: &[f32],
        y: &[f32],
    ) -> Result<Self, String> {
        validate_topology(node_count, tail, head, x, y)?;
        let node_count_u32 = u32::try_from(node_count)
            .map_err(|_| format!("CCH node count {node_count} does not fit u32"))?;
        let order = compute_order_inertial(node_count_u32, tail, head, x, y);
        if order.len() != node_count {
            return Err(format!(
                "CCH order has {} nodes but topology has {node_count}",
                order.len()
            ));
        }
        let cch = CCH::new(&order, tail, head, |_| {}, false);
        Ok(Self {
            cch,
            node_count,
            arc_count: tail.len(),
            order,
        })
    }

    pub(crate) fn customize<'a>(&'a self, arc_weights: &[u32]) -> Result<CchMetric<'a>, String> {
        validate_weights(arc_weights, self.arc_count)?;
        Ok(CchMetric {
            inner: RoutingMetric::new(&self.cch, arc_weights.to_vec()),
            node_count: self.node_count,
        })
    }

    pub(crate) fn order(&self) -> &[u32] {
        &self.order
    }
}

impl CchMetric<'_> {
    pub(crate) fn new_query(&self) -> CchReusableQuery<'_> {
        CchReusableQuery {
            inner: CCHQuery::new(&self.inner),
            node_count: self.node_count,
        }
    }
}

impl CchReusableQuery<'_> {
    pub(crate) fn shortest_path(
        &mut self,
        sources: &[(u32, u32)],
        targets: &[(u32, u32)],
    ) -> Result<CchPath, String> {
        validate_endpoints(sources, self.node_count, "source")?;
        validate_endpoints(targets, self.node_count, "target")?;

        for &(node, offset) in sources {
            self.inner.add_source(node, offset);
        }
        for &(node, offset) in targets {
            self.inner.add_target(node, offset);
        }
        let result = self.inner.run();
        let distance = result
            .distance()
            .ok_or_else(|| "CCH query is unreachable".to_string())?;
        let node_path = result
            .node_path()
            .into_iter()
            .map(|node| node as usize)
            .collect();
        let arc_path = result
            .arc_path()
            .into_iter()
            .map(|arc| arc as usize)
            .collect();
        Ok(CchPath {
            distance,
            node_path,
            arc_path,
        })
    }
}

fn validate_topology(
    node_count: usize,
    tail: &[u32],
    head: &[u32],
    x: &[f32],
    y: &[f32],
) -> Result<(), String> {
    if node_count == 0 || node_count > u32::MAX as usize {
        return Err(format!(
            "CCH node count {node_count} must be in 1..={}",
            u32::MAX
        ));
    }
    if tail.len() != head.len() {
        return Err(format!(
            "CCH arc arrays must have equal length: tail={}, head={}",
            tail.len(),
            head.len()
        ));
    }
    if x.len() != node_count || y.len() != node_count {
        return Err(format!(
            "CCH coordinate arrays do not match {node_count} nodes: x={}, y={}",
            x.len(),
            y.len()
        ));
    }
    if let Some((node, (&x, &y))) = x
        .iter()
        .zip(y)
        .enumerate()
        .find(|(_, (x, y))| !x.is_finite() || !y.is_finite())
    {
        return Err(format!(
            "CCH node {node} has non-finite coordinates ({x}, {y})"
        ));
    }
    for (arc, (&tail_node, &head_node)) in tail.iter().zip(head).enumerate() {
        if tail_node as usize >= node_count || head_node as usize >= node_count {
            return Err(format!(
                "CCH arc {arc} endpoint out of bounds for {node_count} nodes: {tail_node}->{head_node}"
            ));
        }
    }
    Ok(())
}

fn validate_weights(weights: &[u32], expected: usize) -> Result<(), String> {
    if weights.len() != expected {
        return Err(format!(
            "CCH metric has {} arc weights but topology has {expected} arcs",
            weights.len()
        ));
    }
    if let Some((arc, weight)) = weights
        .iter()
        .copied()
        .enumerate()
        .find(|(_, weight)| *weight == 0 || *weight >= CCH_INFINITY)
    {
        return Err(format!("CCH arc {arc} has invalid weight {weight}"));
    }
    Ok(())
}

fn validate_endpoints(
    endpoints: &[(u32, u32)],
    node_count: usize,
    kind: &str,
) -> Result<(), String> {
    if endpoints.is_empty() {
        return Err(format!("CCH query has no {kind} states"));
    }
    if let Some((node, _)) = endpoints
        .iter()
        .copied()
        .find(|(node, _)| *node as usize >= node_count)
    {
        return Err(format!(
            "CCH {kind} node {node} is out of bounds for {node_count} nodes"
        ));
    }
    if let Some((_, offset)) = endpoints
        .iter()
        .copied()
        .find(|(_, offset)| *offset >= CCH_INFINITY)
    {
        return Err(format!(
            "CCH {kind} offset {offset} reaches the infinity sentinel"
        ));
    }
    Ok(())
}
