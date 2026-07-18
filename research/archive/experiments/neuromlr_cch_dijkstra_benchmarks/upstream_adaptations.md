# NeuroMLR upstream and compatibility adaptations

The official checkout is left byte-for-byte clean at commit
`c45e3b5811e5a59b36e4682307d2196c02dac360`; model classes are imported from
its `model_all.py` and `models_general.py`. The tracked driver is
the retired `scripts/neuromlr_fair.py` adapter. Recover that exact source with
`git show 0f4bd55:scripts/neuromlr_fair.py` when auditing the old run.

The following execution adaptations are required for the frozen comparison:

1. The upstream `train.py` imports a missing `model.py`. The adapter imports
   the present upstream `model_all.Model`; no layer dimensions or forward
   formula are changed.
2. Upstream training loads full test and fixed test before the first update and
   evaluates test after training. The adapter accepts only explicit common
   train and validation paths during training and has no test argument in that
   mode.
3. Upstream `condense_edges()` maps all parallel raw road IDs sharing `(u,v)`
   to one ID. That would alter 40,065 selected common training trajectories.
   The adapter uses the shapefile record index one-to-one and exports the same
   raw IDs as the project.
4. The upstream loader removes loops in train but not necessarily in
   validation/test. Both methods instead receive the already-frozen common
   filter: at least five roads, continuity, and drop on any repeated original
   graph node. No method-specific loop repair is performed.
5. Upstream code does not seed Python, NumPy, Torch, or CUDA. The adapter fixes
   all seeds to `20260716`, disables cuDNN benchmarking, and records the 128
   Lipschitz anchors. PyG CUDA scatter reductions are not claimed to be
   bitwise reproducible across GPU/driver versions; this is one fixed run, not
   a multi-seed uncertainty estimate.
6. The published script builds its road vocabulary after unconditionally
   loading test. The adapter represents every raw map road and every map node,
   preserving the inductive endpoint-node model while avoiding test access.
7. Upstream metrics weight roads by physical length. They are retained only as
   supplementary output. Primary results use the shared raw-road macro
   evaluator.
8. Upstream whole-object `torch.save(model)` is fragile across Python/PyTorch
   versions. The adapter saves the unchanged model's state dict, upstream
   commit, model configuration, optimizer state, seed, epoch, and graph hash.
9. Python 3.8 / Torch 1.6 / CUDA 10.2 cannot target the RTX 5060 Ti. The
   compatibility environment uses Python 3.12.3, Torch 2.7.1+cu128, and PyG
   2.6.1. Basic `GCNConv` and MLP semantics used by the upstream classes are
   unchanged.
10. Lipschitz initialization uses a SciPy sparse reverse graph for modern
    compatibility. Parallel roads are explicitly coalesced with minimum
    haversine cost, matching NetworkX multigraph Dijkstra; sparse duplicate
    summation is forbidden. A two-epoch development run that used summed
    duplicates was stopped before checkpoint selection and excluded.
11. Greedy generation retains the upstream maximum of 300 transitions and its
    closest-point truncation when the destination is not reached. The adapter
    does not append the true destination road or otherwise repair a failed
    rollout; endpoint failures are preserved in the raw predictions.

The formal model still uses GCN, one configured GNN layer (which creates the
same two upstream `GCNConv` transforms), 128-dimensional Lipschitz features,
three hidden MLP layers of width 256, Adam AMSGrad at learning rate 0.001,
batch size 32, 50 epochs, no traffic features, and exactly `L-1` next-road
targets for a length-`L` complete route.

The formal external baseline is NeuroMLR-Greedy only. NeuroMLR-Dijkstra support
in the adapter is dormant and is not used for validation selection, test
quality, or timing, following the final scope decision before protocol freeze.
