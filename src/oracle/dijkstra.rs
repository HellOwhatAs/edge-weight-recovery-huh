#[derive(Clone, Debug, PartialEq)]
pub struct DijkstraPath {
    pub distance: f64,
    pub original_edges: Vec<usize>,
}

/// Exact dense Dijkstra for small correctness and continuous-f64 checks.
///
/// This intentionally uses an O(|V||E|) implementation. Production routing
/// belongs to the CCH oracle; this routine stays simple enough to audit.
pub fn shortest_path_f64(
    node_count: usize,
    tail: &[u32],
    head: &[u32],
    weights: &[f64],
    source: u32,
    target: u32,
) -> Result<Option<DijkstraPath>, String> {
    if tail.len() != head.len() || tail.len() != weights.len() {
        return Err(format!(
            "Dijkstra edge-array length mismatch: tail={}, head={}, weights={}",
            tail.len(),
            head.len(),
            weights.len()
        ));
    }
    if source as usize >= node_count || target as usize >= node_count {
        return Err(format!(
            "Dijkstra OD ({source}, {target}) is out of bounds for {node_count} nodes"
        ));
    }
    for (edge, ((&tail_node, &head_node), &weight)) in
        tail.iter().zip(head).zip(weights).enumerate()
    {
        if tail_node as usize >= node_count || head_node as usize >= node_count {
            return Err(format!(
                "Dijkstra edge {edge} endpoint out of bounds: {tail_node}->{head_node}"
            ));
        }
        if !weight.is_finite() || weight < 0.0 {
            return Err(format!(
                "Dijkstra edge {edge} has invalid nonnegative f64 weight {weight}"
            ));
        }
    }

    let source = source as usize;
    let target = target as usize;
    let mut distance = vec![f64::INFINITY; node_count];
    let mut settled = vec![false; node_count];
    let mut predecessor = vec![None::<(usize, usize)>; node_count];
    distance[source] = 0.0;

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
        if node == target {
            break;
        }

        for edge in 0..tail.len() {
            if tail[edge] as usize != node {
                continue;
            }
            let next_node = head[edge] as usize;
            if settled[next_node] {
                continue;
            }
            let candidate = distance[node] + weights[edge];
            if candidate < distance[next_node] {
                distance[next_node] = candidate;
                predecessor[next_node] = Some((node, edge));
            }
        }
    }

    if !distance[target].is_finite() {
        return Ok(None);
    }
    let mut original_edges = Vec::new();
    let mut node = target;
    while node != source {
        let Some((previous, edge)) = predecessor[node] else {
            return Err("Dijkstra predecessor chain is incomplete".to_string());
        };
        original_edges.push(edge);
        node = previous;
    }
    original_edges.reverse();
    Ok(Some(DijkstraPath {
        distance: distance[target],
        original_edges,
    }))
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
        assert_eq!(result.original_edges, vec![2, 3]);
    }

    #[test]
    fn reports_unreachable_and_rejects_invalid_weights() {
        assert_eq!(
            shortest_path_f64(3, &[0], &[1], &[1.0], 0, 2).unwrap(),
            None
        );
        assert!(shortest_path_f64(2, &[0], &[1], &[f64::NAN], 0, 1).is_err());
        assert!(shortest_path_f64(2, &[0], &[1], &[-1.0], 0, 1).is_err());
    }
}
