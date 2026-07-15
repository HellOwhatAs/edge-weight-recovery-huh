#[derive(Clone, Debug, PartialEq)]
pub struct DijkstraPath {
    pub distance: f64,
    pub node_path: Vec<usize>,
    pub arc_path: Vec<usize>,
}

/// Exact dense Dijkstra with arbitrary nonnegative source and target offsets.
///
/// This intentionally uses an O(|V||E|) implementation. Production routing
/// belongs to CCH; this routine stays simple enough to audit and is used as a
/// correctness reference for both ordinary arc-weight graphs and transformed
/// node-weight graphs.
pub fn shortest_path_multi_source_f64(
    node_count: usize,
    tail: &[u32],
    head: &[u32],
    weights: &[f64],
    sources: &[(u32, f64)],
    targets: &[(u32, f64)],
) -> Result<Option<DijkstraPath>, String> {
    validate_inputs(node_count, tail, head, weights, sources, targets)?;

    let mut distance = vec![f64::INFINITY; node_count];
    let mut settled = vec![false; node_count];
    let mut predecessor = vec![None::<(usize, usize)>; node_count];
    for &(source, offset) in sources {
        let source = source as usize;
        if offset < distance[source] {
            distance[source] = offset;
            predecessor[source] = None;
        }
    }

    for _ in 0..node_count {
        let next = (0..node_count)
            .filter(|&node| !settled[node])
            .min_by(|&left, &right| {
                distance[left]
                    .total_cmp(&distance[right])
                    .then_with(|| left.cmp(&right))
            });
        let Some(node) = next else {
            break;
        };
        if !distance[node].is_finite() {
            break;
        }
        settled[node] = true;

        for arc in 0..tail.len() {
            if tail[arc] as usize != node {
                continue;
            }
            let next_node = head[arc] as usize;
            if settled[next_node] {
                continue;
            }
            let candidate = distance[node] + weights[arc];
            if candidate < distance[next_node] {
                distance[next_node] = candidate;
                predecessor[next_node] = Some((node, arc));
            }
        }
    }

    let selected_target = targets
        .iter()
        .copied()
        .filter_map(|(node, offset)| {
            let total = distance[node as usize] + offset;
            total.is_finite().then_some((node as usize, total))
        })
        .min_by(|(left_node, left_cost), (right_node, right_cost)| {
            left_cost
                .total_cmp(right_cost)
                .then_with(|| left_node.cmp(right_node))
        });
    let Some((target, total_distance)) = selected_target else {
        return Ok(None);
    };

    let mut node_path = vec![target];
    let mut arc_path = Vec::new();
    let mut node = target;
    while let Some((previous, arc)) = predecessor[node] {
        arc_path.push(arc);
        node = previous;
        node_path.push(node);
    }
    node_path.reverse();
    arc_path.reverse();

    Ok(Some(DijkstraPath {
        distance: total_distance,
        node_path,
        arc_path,
    }))
}

/// Exact dense single-source/single-target Dijkstra.
pub fn shortest_path_f64(
    node_count: usize,
    tail: &[u32],
    head: &[u32],
    weights: &[f64],
    source: u32,
    target: u32,
) -> Result<Option<DijkstraPath>, String> {
    shortest_path_multi_source_f64(
        node_count,
        tail,
        head,
        weights,
        &[(source, 0.0)],
        &[(target, 0.0)],
    )
}

fn validate_inputs(
    node_count: usize,
    tail: &[u32],
    head: &[u32],
    weights: &[f64],
    sources: &[(u32, f64)],
    targets: &[(u32, f64)],
) -> Result<(), String> {
    if node_count == 0 {
        return Err("Dijkstra requires at least one node".to_string());
    }
    if tail.len() != head.len() || tail.len() != weights.len() {
        return Err(format!(
            "Dijkstra edge-array length mismatch: tail={}, head={}, weights={}",
            tail.len(),
            head.len(),
            weights.len()
        ));
    }
    if sources.is_empty() || targets.is_empty() {
        return Err("Dijkstra requires at least one source and one target".to_string());
    }
    for (arc, ((&tail_node, &head_node), &weight)) in tail.iter().zip(head).zip(weights).enumerate()
    {
        if tail_node as usize >= node_count || head_node as usize >= node_count {
            return Err(format!(
                "Dijkstra arc {arc} endpoint out of bounds: {tail_node}->{head_node}"
            ));
        }
        if !weight.is_finite() || weight < 0.0 {
            return Err(format!(
                "Dijkstra arc {arc} has invalid nonnegative f64 weight {weight}"
            ));
        }
    }
    for (kind, endpoints) in [("source", sources), ("target", targets)] {
        for &(node, offset) in endpoints {
            if node as usize >= node_count {
                return Err(format!(
                    "Dijkstra {kind} node {node} is out of bounds for {node_count} nodes"
                ));
            }
            if !offset.is_finite() || offset < 0.0 {
                return Err(format!(
                    "Dijkstra {kind} node {node} has invalid nonnegative offset {offset}"
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_exact_f64_shortest_path() {
        let result =
            shortest_path_f64(4, &[0, 1, 0, 2], &[1, 3, 2, 3], &[5.0, 5.0, 2.0, 2.0], 0, 3)
                .unwrap()
                .unwrap();
        assert_eq!(result.distance, 4.0);
        assert_eq!(result.node_path, vec![0, 2, 3]);
        assert_eq!(result.arc_path, vec![2, 3]);
    }

    #[test]
    fn supports_multiple_sources_and_target_offsets() {
        let result = shortest_path_multi_source_f64(
            4,
            &[0, 1, 2],
            &[1, 3, 3],
            &[2.0, 2.0, 1.0],
            &[(0, 10.0), (2, 3.0)],
            &[(1, 8.0), (3, 0.0)],
        )
        .unwrap()
        .unwrap();
        assert_eq!(result.distance, 4.0);
        assert_eq!(result.node_path, vec![2, 3]);
        assert_eq!(result.arc_path, vec![2]);
    }

    #[test]
    fn reports_unreachable_and_rejects_invalid_weights() {
        assert_eq!(
            shortest_path_f64(3, &[0], &[1], &[1.0], 0, 2).unwrap(),
            None
        );
        assert!(shortest_path_f64(2, &[0], &[1], &[f64::NAN], 0, 1).is_err());
        assert!(shortest_path_f64(2, &[0], &[1], &[-1.0], 0, 1).is_err());
        assert!(shortest_path_multi_source_f64(2, &[0], &[1], &[1.0], &[], &[(1, 0.0)]).is_err());
    }
}
