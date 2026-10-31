## 3. Existing solutions and their limits

This section describes method *families* in general terms. Where a specific paper is named, it was checked (URLs in the reference list at the end of section 14); where a claim about a family rests on common knowledge that we did not verify against a specific source, it is marked [citation needed]. We did not run any of these methods ourselves (section 10.3).

### 3.1 Density and community methods

*What they do.* Partition the graph into communities [citation needed] or find dense subgraphs, then score groups that are unusually tightly connected, on the idea that colluding accounts trade mostly with each other. A verified example of dense-subgraph fraud detection is Fraudar (Hooi et al., KDD 2016, extended in ACM TKDD 2017, ref. 6), which searches for dense subgraphs while resisting camouflage.

*What they miss and why.* A ring of three accounts has three edges: it is not dense in any useful sense, and a ring spread across a heavy-tailed graph is not a community. Density is a statement about how many edges exist among a set of nodes; a loop is a statement about the *direction* in which the few edges point. A dense blob of one-way traffic scores high and is harmless; a sparse loop scores low and is the target.

### 3.2 Centrality scores

*What they do.* Score each node by how central it is (degree, betweenness, PageRank and relatives) and flag accounts that are unusually central or unusually peripheral. [citation needed]

*What they miss and why.* Centrality describes a node's position, not whether value circulates back. A ring member can have ordinary centrality. We still use degrees, betweenness and clustering as the **B baseline features** (section 4.8), because they are the honest comparison: if topology cannot beat them, it is not adding value.

### 3.3 Graph neural networks (GNNs)

*What they do.* Learn node, edge or subgraph representations by passing messages along edges and train a classifier on labelled data. The AMLworld paper (Altman et al., NeurIPS 2023, Datasets and Benchmarks track, ref. 1) released the IBM data we use and ran baselines with graph neural networks and gradient-boosted trees (section 5.2). Weber et al. (KDD 2019 workshop, ref. 7) compared logistic regression, random forest, multilayer perceptrons and graph convolutional networks on the Elliptic Bitcoin transaction graph and reported that random forest performed best.

*What they miss and why.* This is not a defect of GNNs in general, but of using them here: they need substantial labelled data and tuning, are hard to interpret ("which loop, how strong?"), and our design rules out deep learning (classical methods only), so **we did not compare against a GNN**. We cannot claim to beat one (section 10.3).

### 3.4 Cycle enumeration and strongly connected components

*What they do.* Find strongly connected components (groups where every node can reach every other) and list simple directed cycles up to a length bound. Within the length bound and the sparsified graph, every account-level ring is enumerated exactly; recall against known rings is partial (section 8.7), and the bounded-enumeration algorithm has a published caveat (section 11.1). According to the networkx documentation (ref. 13), `simple_cycles` uses a nonrecursive, iterator version of Johnson's algorithm (Johnson 1975, ref. 12) when no length bound is given, and a version of the algorithm of Gupta and Suzumura (ref. 10) when `length_bound` is given, which is our case.

*What they miss and why.* The answer is **binary**: a cycle either exists or it does not. There is no measure of how strong it is (how suspicious are its edges), how stable it is as thresholds change, or whether its neighbourhood fills, splits or crowds it. Lots of benign loops exist too (arbitrage, market making, exchange hot wallets), so enumeration alone has a false-positive problem. In *our* pipeline cycle enumeration is kept as the localiser (section 4.7) and the new information we test is what persistence adds on top.

### 3.5 Persistent homology on blockchain graphs

*What it does.* Ofori-Boateng, Segovia Dominguez, Kantarcioglu, Akcora and Gel ("Topological Anomaly Detection in Dynamic Multilayer Blockchain Networks", ECML PKDD 2021, arXiv:2106.01806) use *clique* persistent homology to track the shape of multilayer transaction graphs over time, introduce a "stacked persistence diagram" and prove it stable under small perturbations. They report results on Ethereum and the Ripple Credit Network.

*What it misses (for our purpose).* As the v3 proposal puts it, that line of work flags that the network's *global shape changed*; it does not isolate an individual ring, type rings, or produce a per-ring strength. Two technical points matter here. Clique persistent homology is defined on an *undirected* clique complex, and in an undirected clique complex every triangle is filled, so a circular triangle A to B to C to A is indistinguishable from a transitive one (section 4.5). We did not run the undirected ablation (section 11.4), so this difference is argued, not measured.

### 3.6 Fixed-threshold alarming

*What it does.* Flag a candidate when its score passes a cutoff tuned on history.

*What it misses and why.* Tactics drift (ring timing becomes more irregular, amounts noisier, decoy trades appear), so the best cutoff moves. The optimization report in this repository measures the damage on synthetic drift: a threshold frozen after the first windows earns about a quarter of the hindsight-optimal reward (section 8.1, 8.9).

### 3.7 Summary table

| Family | Finds loops? | Strength score? | Typing? | Uses direction? | Why it falls short for rings |
|---|---|---|---|---|---|
| Density / community | No (finds blobs) | density | No | usually no | sparse loops are not dense |
| Centrality | No | node centrality | No | partly | position, not circulation |
| GNN | Learned | learned score | possible | yes | needs labels; opaque; not tested here |
| Cycle enumeration / SCC | Yes, exactly | none (binary) | No | yes | no strength or stability; many benign loops |
| Clique persistent homology | Global shape only | global | No | no (undirected) | does not isolate rings |
| Fixed threshold | n/a | n/a | n/a | n/a | decays under drift |
