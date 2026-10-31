
## 11. Limitations, deviations from v3, and what was not done

### 11.1 Limitations, each tied to a number

<<BLOCK 7>>

Two more limits belong in every discussion of the results.

* **Single generator.** Synthetic results come from one generator whose designer chose the scenarios (generator circularity); the IBM data come from one simulator.
* **Bounded cycle enumeration.** Our candidates use networkx `simple_cycles` with a length bound, which per its documentation applies a version of the Gupta and Suzumura algorithm (ref. 10). A 2025 preprint (ref. 11) reports a counterexample in which that algorithm misses valid cycles; we did not test our enumeration against it, so candidate recall could be slightly optimistic or pessimistic for reasons unrelated to tau and `max_len`.
* **Six seeds.** Synthetic statistics use 6 seeds (the v3 requirement is at least 5); the bandit benchmark uses 6 stream seeds.

### 11.2 Deviations from the v3 proposal

<<BLOCK 9>>

### 11.3 Where the code and the documents disagree (the code is followed)

| Item | Document says | Code does | Resolution |
|---|---|---|---|
| Timing-regularity imputation | v3 section 3.1: a pair with a *single* transaction gets the window median `s_reg` | `_gap_cv` needs at least 3 transactions (2 gaps); **pairs with 1 or 2 transactions are imputed**. Weights use the window median of `s_reg`; D features use the window median of the defined gap CVs (a different fill) | Followed the code; documented as a deviation |
| De-duplication rule | v3 section 3.3: keep the lowest-birth near-duplicate | shortest first, then birth; nothing removed for `max_len` at most 9 | Followed the code; documented |
| Fold counts | README said 58 of 60 folds have fewer than 5 positive groups | `ibm_split_counts.csv` gives 144 of 150 fold-repeats (50 of 50 at tau 0.05) | README corrected |
| Betweenness sparsity | README said k = 50 gave 166 non-zero nodes | only k = 100 caches exist; they give a non-zero share of 0.1594 on HI window 0 | Claim removed |
| Recall of IBM cycle attempts | README: 8/8/8/9 recovered | account-set matches reproduce 8/8/8/9; requiring a positive CYCLE label gives fewer | Both reported, gap stated |
| Bandit status | `paper/paper_content.md` section 5.10 still carries a `[TODO-CODE]` saying the bandits exist only in the proposal | `threshold/bandit_sim.py` exists and was run | **Not edited** (that file was only partly updated); the explainer follows the code |
| Status lists | `architecture.md`, `prd.md` and the README status lists were stale after the experiments | tests and modules exist | Updated in separate commits |

### 11.4 What was not done, and why

| Item | Why |
|---|---|
| **Phase 5: NFT experiment** | Running queries on Dune is blocked on the free tier. BigQuery access was confirmed and the schemas inspected. `nft.wash_trades` combines five filters; filters 2 and 3 are cycle-type rules, so the leakage-controlled subset would be filters 4 and 5 only. |
| Pump-and-dump on real data | Only a synthetic, no-topology track exists (section 8.14); no real pump-and-dump data were used. |
| Undirected clique-complex ablation | Not run; the directed-versus-undirected difference is therefore argued, not measured. |
| Persistence-image / vectorization ablations | Not run; only landscapes and scalar summaries exist. |
| Classifiers on per-score filtrations | Not run. |
| Sweeps over window length, stride, label threshold X, `max_len` (in classification) and tau beyond 0.05 | Only candidate recall was swept over tau and `max_len`; classification was run at IBM tau 0.05 only. |
| Severity on real data | IBM has no next-window wash volume. |
| A comparison against a GNN | Classical methods only by design. |
