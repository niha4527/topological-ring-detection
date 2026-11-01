# sources.md

Folder: `paper/Reference papers/` (git-ignored). Summaries are in my own words from the abstract and introduction of each PDF (first pages; two image-only PDFs read visually on pages 1-2). Page numbers are PDF page indices. Results quoted as the papers report them; none were re-run.

**Inventory.** 19 PDF files, 17 distinct papers. Duplicates: `s41598-024-75348-7 (1).pdf` is byte-identical to `s41598-024-75348-7.pdf`; `2306.16424v3.pdf` (arXiv) and the `NeurIPS-2023-...pdf` (proceedings) are the same Altman et al. paper. Image-only (no text layer, read visually): `Detecting and Quantifying Wash Trading...pdf`, `Subgraph anomaly detection in financial transaction networks.pdf`. No filename/content mismatch found. Metadata not printed in the PDF are marked [VERIFY] in `refs.bib`.

| key | what it is | venue / year | metadata source |
|---|---|---|---|
| weber2019aml | Elliptic data, LR/RF/MLP/GCN on Bitcoin | KDD'19 Workshop on Anomaly Detection in Finance | PDF p.1; no DOI printed |
| ofori2021topological | Clique persistent homology, stacked PD | arXiv v2 (ECML PKDD 2021 per explainer) | PDF p.1; venue [VERIFY] |
| liu2023nft | NFT wash-trade rule on 7 collections | arXiv 2023 preprint | PDF p.1 |
| altman2023realistic | AMLworld generator and datasets | NeurIPS 2023 D&B | PDF footer; arXiv 2306.16424 |
| vivek2025rosfd | Online streaming fraud detection with drift detectors | arXiv 2025 preprint | PDF p.1; year [VERIFY] |
| lutgehetmann2020flagser | flagser | Algorithms 13(1):19, 2020 | PDF p.1, DOI 10.3390/a13010019 |
| victor2021wash | DEX wash trading | WWW'21 | PDF p.1, DOI 10.1145/3442381.3449824 |
| cao2016wash | Digraph + dynamic programming wash trade | IEEE TNNLS 27(11):2351-2363, 2016 | PDF p.1, DOI 10.1109/TNNLS.2015.2480959 |
| ouyang2024bitchetg | Bit-CHetG subgraph contrastive learning | Entropy 26(3):211, 2024 | PDF p.1 |
| zhao2019wkpi | Learned weights for persistence kernels | NeurIPS 2019 | PDF p.1; volume [VERIFY] |
| chen2023parallel | Parallel path detection in bank graphs | PeerJ CS 9:e1749, 2023 | PDF p.1 |
| tosic2025beyond | NFT wash trading with Ethereum network | Financial Innovation 11:86, 2025 | PDF p.1 |
| govc2021tournaplex | Tournaplexes, directionality filtrations | J. Appl. Comput. Topol. 5:313-337, 2021 | PDF p.1 |
| jensen2023synthaml | SynthAML | Scientific Data 10:661, 2023 | PDF p.1 |
| arnold2024motifs | Temporal motifs in crypto networks | Sci. Reports 14:26569, 2024 | PDF p.1 |
| sirolly2025network | Network-based wash trading detection | SSRN 5714122, 6 Nov 2025 (working paper) | PDF p.1 |
| pei2020sade | SADE subgraph anomaly detection | ICAIF'20 | PDF p.1, DOI 10.1145/3383455.3422548 |

## Summaries

**weber2019aml.** *Problem:* anti-money-laundering on Bitcoin with open data. *Method:* releases the Elliptic dataset (over 200K transaction nodes, 234K directed payment flows, 166 features) and compares logistic regression, random forest, MLP and graph convolutional networks on illicit-transaction classification. *Result:* random forest does best; GCN is not superior. *Relation:* shows a tree/linear baseline can beat a graph neural network on transaction graphs; supports our choice of classical models; not wash trading.

**ofori2021topological.** *Problem:* anomalies in dynamic multilayer blockchain graphs. *Method:* clique persistent homology, a stacked persistence diagram for multilayer networks, a stability proof. *Result:* reports gains over state-of-the-art on Ethereum and Ripple data. *Relation:* closest TDA-on-transactions work; it tracks the shape of the whole network on an undirected clique complex, while we classify individual candidate rings with a directed complex. We did not run it.

**liu2023nft.** *Problem:* wash trading in NFT markets. *Method:* flags a wash trade when a wallet repurchases a token within 30 days of selling it, and finds intermediate transactions in the cycle (BFS). *Result:* on 7 collections, on average 0.14% of transactions, 0.11% of wallets and 0.16% of tokens are involved. *Relation:* a rule-based cycle heuristic; this kind of rule is what makes NFT proxy labels leak cycle structure (our reason for the planned filter subset).

**altman2023realistic.** *Problem:* no public, realistic AML data with complete labels. *Method:* agent-based generator calibrated to real transactions, plus public datasets (HI/LI, Small/Medium/Large) with labelled laundering patterns. *Result:* argues synthetic ground truth is complete while real labels miss undetected laundering. *Relation:* our IBM HI-Small and LI-Small data come from here; we state that IBM is synthetic AML data, not market data.

**vivek2025rosfd.** *Problem:* fraud detection in streams with concept drift. *Method:* offline-initialised incremental model, drift detectors (DDM, EDDM, ADWIN) decide when to retrain. *Result:* the "train only when required" strategy cuts retrains with little AUC loss; ADWIN best, Adaptive Random Forest best model. *Relation:* supports framing drift adaptation as retrain-versus-keep; our threshold study is a different, simulated setting.

**lutgehetmann2020flagser.** *Problem:* computing persistent homology of directed flag complexes at scale. *Method:* flagser software built on Ripser, parallel complex construction, approximate option. *Result:* applied to Blue Brain microcircuits. *Relation:* our persistence engine (via pyflagser). Returns no representative cycles, which is why we localise by cycle enumeration.

**victor2021wash.** *Problem:* wash trading on order-book DEXs. *Method:* finds account sets and trade structures meeting legal wash-trade definitions on IDEX and EtherDelta. *Result:* lower bound of about 159 million USD wash-traded, more than 30% of tokens affected; most structures use one or two accounts. *Relation:* motivates the problem and shows most real structures are self-trades and 2-cycles, so our longer rings are a harder, rarer case.

**cao2016wash.** *Problem:* wash trades from matched orders in stock markets. *Method:* models collusion as a directed trader graph, finds suspicious matched orders and then collusive groups, each solved as a simplified knapsack by dynamic programming. *Result:* detects all primary wash-trade scenarios on seven NASDAQ and LSE datasets. *Relation:* an earlier directed-graph wash-trade method on order data; ours uses settled transfers and a learned classifier.

**ouyang2024bitchetg.** *Problem:* money-laundering group detection (not single transactions) on Bitcoin. *Method:* metapath subgraphs, GNN embeddings, supervised contrastive learning (Bit-CHetG). *Result:* Micro-F1 at least 5% above others on two real datasets. *Relation:* a GNN family member for subgraph-level detection; we did not compare against it.

**zhao2019wkpi.** *Problem:* fixed weights in persistence vectorisations. *Method:* a weighted persistence-image kernel (WKPI) with learned weights, trained for graph classification. *Result:* matches or beats prior graph-classification methods on benchmarks. *Relation:* shows vectorisation choices matter; we used fixed landscapes and did not run a persistence-image or learned-weight ablation.

**chen2023parallel.** *Problem:* find suspicious accounts in large bank transaction graphs. *Method:* directed graph, shrink SCCs, parallel DFS path detection. *Result:* scales on multicore machines against serial baselines. *Relation:* same SCC-then-search localisation idea as ours, with no strength score and no learned classifier.

**tosic2025beyond.** *Problem:* NFT wash trading hidden behind intermediary transfers. *Method:* joins NFT ownership traces with the full Ethereum transaction network. *Result:* on 7 collections wash trading is up to 24% of volume (93% for Meebits). *Relation:* real-data evidence of how common wash trading is on NFTs, and where our unrun NFT experiment would have gone.

**govc2021tournaplex.** *Problem:* directed flag complexes lose information about non-transitive cliques. *Method:* tournaplexes and directionality filtrations. *Result:* finer discrimination of graph dynamics on Blue Brain data. *Relation:* a directed-topology variant we did not use; supports that direction carries topological information.

**jensen2023synthaml.** *Problem:* no public bank AML data. *Method:* SynthAML, 20,000 alerts and over 16 million transactions generated from real Spar Nord data. *Result:* reports that performance on it transfers to the real world. *Relation:* an alternative synthetic AML benchmark; we did not use it.

**arnold2024motifs.** *Problem:* temporal motifs in crypto transaction graphs. *Method:* counts three-transaction, up-to-three-user motifs per user and over time (Raphtory). *Result:* whole-network counts can mislead; per-user and per-period views reveal events and anomalous activity. *Relation:* supports our time-windowed, local view; related to the pump-and-dump motif track.

**sirolly2025network.** *Problem:* wash trading in exchange data. *Method:* iterative network procedure assuming wash traders form nearly closed clusters of colluding counterparties. *Result:* on Polymarket, wash-like patterns peak near 60% of volume in December 2024. *Relation:* the closest recent network wash-trade detector; it uses closed-cluster structure, not directed persistence. Working paper.

**pei2020sade.** *Problem:* anomalous subgraphs, not nodes, in financial networks. *Method:* SADE: role-guided subgraph embedding plus Isolation Forest. *Result:* effective on synthetic and real networks (as reported). *Relation:* subgraph-level detection that ignores edge direction and weights (undirected, unweighted by design); contrast with ours.

## Not in the PDF folder, carried over from explainer 14.3 (unchecked against PDFs)
Garivier and Moulines (discounted UCB), Trovò et al. and Fiandri et al. (sliding-window Thompson), Bubenik (landscapes), Hooi et al. (Fraudar), Gupta and Suzumura and Bauernöppel and Sack (bounded-cycle enumeration), Johnson (cycle enumeration), networkx docs. All marked [VERIFY] in `refs.bib`. Community-detection and centrality claims stay [citation needed].
