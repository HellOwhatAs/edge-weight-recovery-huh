use crate::utils::build_pb;
use indicatif::{MultiProgress, ProgressIterator};
use rayon::prelude::*;
use routingkit_cch::shp_utils;
use routingkit_cch::{CCHMetric, CCHQuery};
use std::collections::HashMap;
use std::collections::HashSet;

/// Load graph data and preprocessed trips from shapefiles and pickle files.
pub fn load_graph(
    city: &str,
) -> (
    Vec<u32>,
    Vec<u32>,
    Vec<u32>,
    Vec<f32>,
    Vec<f32>,
    Vec<((u32, u32), Vec<usize>)>,
    Vec<((u32, u32), Vec<usize>)>,
) {
    let (Ok(edges), Ok(nodes)) = (
        shp_utils::load_edges(&format!("data/{city}_data/map/edges.shp")),
        shp_utils::load_nodes(&format!("data/{city}_data/map/nodes.shp")),
    ) else {
        panic!("Failed to load data for city: {city}");
    };
    let shp_utils::GraphArrays {
        osmids,
        xs,
        ys,
        tail,
        head,
        weight,
    } = shp_utils::build_graph_arrays(&nodes, &edges).unwrap();
    let osmid2idx = osmids
        .iter()
        .enumerate()
        .map(|(i, &osmid)| (osmid, i as u32))
        .collect::<HashMap<_, _>>();

    let tail = tail.into_iter().map(|x| x as u32).collect::<Vec<u32>>();
    let head = head.into_iter().map(|x| x as u32).collect::<Vec<u32>>();
    let weights = weight
        .into_iter()
        .map(|x| (x * 1e3) as u32)
        .collect::<Vec<u32>>();
    let lat = xs.into_iter().map(|x| x as f32).collect::<Vec<f32>>();
    let lon = ys.into_iter().map(|x| x as f32).collect::<Vec<f32>>();

    let trip = |mode: &str| {
        let deserialized_trips: Vec<(serde_pickle::Value, Vec<usize>, (usize, usize))> =
            serde_pickle::from_reader(
                std::fs::File::open(format!(
                    "data/{city}_data/preprocessed_{mode}_trips_all.pkl"
                ))
                .unwrap(),
                Default::default(),
            )
            .unwrap();
        let paths: Vec<((u32, u32), Vec<usize>)> = deserialized_trips
            .iter()
            .map(|(_, path, _)| {
                let path = &path[1..path.len() - 1]; // Remove source and target nodes
                let (&first_edge_id, &last_edge_id) = (path.first().unwrap(), path.last().unwrap());
                (
                    (
                        osmid2idx[&edges[first_edge_id].u],
                        osmid2idx[&edges[last_edge_id].v],
                    ),
                    path.to_owned(),
                )
            })
            .collect();
        paths
    };

    (tail, head, weights, lat, lon, trip("train"), trip("test"))
}

pub fn compute_trip_cnt(
    paths: &[((u32, u32), Vec<usize>)],
    edge_count: usize,
    num_chunks: usize,
) -> Vec<usize> {
    paths
        .par_chunks(paths.len().max(1) / num_chunks.max(1))
        .map(|chunk| {
            let mut local = vec![0; edge_count];
            for (_, path) in chunk {
                for &edge_id in path.iter() {
                    local[edge_id] += 1usize;
                }
            }
            local
        })
        .reduce(
            || vec![0; edge_count],
            |mut a, b| {
                a.iter_mut().zip(b.iter()).for_each(|(x, &y)| *x += y);
                a
            },
        )
}

pub fn compute_current_counts(
    metric: &CCHMetric,
    paths: &[((u32, u32), Vec<usize>)],
    edge_count: usize,
    num_chunks: usize,
    epoch: u64,
    city: &str,
    m: &MultiProgress,
) -> Vec<usize> {
    let width = (num_chunks.max(1) - 1).to_string().len();
    paths
        .par_chunks(paths.len().max(1) / num_chunks.max(1))
        .enumerate()
        .map(|(chunk_id, chunk)| {
            let mut query = CCHQuery::new(metric);
            let mut local = vec![0; edge_count];
            let pb = build_pb(
                chunk.len() as u64,
                "cyan/blue",
                format!("{city}[{epoch}]-{chunk_id:<width$}"),
                m,
            );
            for &((s, t), _) in chunk.iter().progress_with(pb) {
                query.add_source(s, 0);
                query.add_target(t, 0);
                for &edge_id in query.run().arc_path().iter() {
                    local[edge_id as usize] += 1usize;
                }
            }
            local
        })
        .reduce(
            || vec![0; edge_count],
            |mut a, b| {
                a.iter_mut().zip(b.iter()).for_each(|(x, &y)| *x += y);
                a
            },
        )
}

pub fn compute_precision(
    metric: &CCHMetric,
    paths: &[((u32, u32), Vec<usize>)],
    weights: &[u32],
    num_chunks: usize,
) -> f32 {
    paths
        .par_chunks(paths.len().max(1) / num_chunks.max(1))
        .map(|chunk| {
            let mut query = CCHQuery::new(metric);
            chunk
                .iter()
                .map(|((s, t), path)| {
                    query.add_source(*s, 0);
                    query.add_target(*t, 0);
                    let s: HashSet<usize> = query
                        .run()
                        .arc_path()
                        .into_iter()
                        .map(|x| x as usize)
                        .collect();
                    let u = path
                        .iter()
                        .filter(|x| s.contains(x))
                        .map(|x| weights[*x])
                        .sum::<u32>() as f32;
                    let b = s.iter().map(|&x| weights[x]).sum::<u32>() as f32;
                    if u == 0.0 { 0.0 } else { u / b }
                })
                .sum::<f32>()
        })
        .sum::<f32>()
        / paths.len() as f32
}

pub fn compute_loss(cur_cnt: &[usize], trip_cnt: &[usize]) -> usize {
    cur_cnt
        .iter()
        .zip(trip_cnt.iter())
        .map(|(a, b)| a.max(b) - a.min(b))
        .sum()
}
