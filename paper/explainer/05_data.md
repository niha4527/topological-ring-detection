## 5. Data

Three data sources were planned; two were used.

### 5.1 The synthetic generator (`src/ringtda/data/synthetic.py`)

**Why synthetic data.** Real labelled wash trading is scarce and its labels are heuristic. A generator that *plants* structures gives exact ground truth: we know which accounts form each ring, so every later stage can be tested against a known answer. The price is **generator circularity**: the designer decides what the data look like, so a method can look good simply because it matches the designer's assumptions (section 11.1).

**Background.** Heavy-tailed account activity (a few accounts are very active), irregular timing, lognormal amounts, mostly one-directional traffic. By default 400 accounts and 6000 background transactions over one week.

**Planted structures.** Each planted structure has a pattern name, a scenario, its accounts and edges, and its own transaction rows carrying a `pattern_id`.

| Structure | What is planted | Class |
|---|---|---|
| `wash_ring`, scenario **isolated** | near-equal amounts around a directed cycle (sizes 2, 3, 5) on a regular cadence (one round about every 4 hours), sparse neighbourhood | positive |
| `wash_ring`, scenario **dense** | the same rings embedded among extra accounts with random incidental traffic, so many incidental loops cross the ring | positive |
| `wash_ring`, scenario **chorded** | a longer ring (size 6) plus forward chords; each round takes the full ring or a shortcut, so every node still nets to zero but the loop splits into shorter cycles | positive |
| `transitive_decoy` | a to b, b to c, a to c: regular, similar amounts, one-way | negative (topology-sensitive); has no cycle, so never becomes a candidate |
| `shortcut_filled` | a regular circular triangle plus a fourth account paying all three members, which fills the loop | negative (topology-sensitive): ring-like under enumeration, but its H1 class dies |
| `arbitrage_loop` | three-party loops fired by random price events with a small profit per hop | hard negative (legitimate loop) |
| `market_maker` | a hub trading both ways with clients at random times and sizes | hard negative |
| `hot_wallet` | an exchange hub with many random deposits and withdrawals; in and out nearly equal | hard negative |

Hard negatives can be switched off (`hard_negatives=False`) to give the "without hard negatives" runs.

**Drift and evasion knobs.** `ring_jitter` (timing noise as a fraction of the period), `amount_noise`, `wash_decoy_frac` (decoy trades to outsiders) and `mm_regularity`. The drift benchmark (section 8.9) raises jitter and amount noise over the stream.

**The dense scenario mostly gets filled (a finding about the generator).** In the dense scenario incidental traffic can fill the planted ring's loop, so its H1 class dies. With the default setting a planted dense ring is still labelled positive, because the label comes from the planted structure, not from topology. `dense_internal_chords=False` removes incidental edges *between members of the same ring*, leaving filling only by outside accounts. We report dense results under both settings (section 8.6).

**Reproducibility.** The default random stream is unchanged by later additions (digests in `tests/test_synthetic_scenarios.py`), and the optimization report's generator is pinned (dense, chorded, shortcut-filled and hot-wallet counts set to 0) so its numbers never change.

### 5.2 IBM AMLworld (HI-Small and LI-Small)

**What it is.** Altman, Blanuša, von Niederhäusern, Egressy, Anghel and Atasu, "Realistic Synthetic Financial Transactions for Anti-Money Laundering Models" (NeurIPS 2023, Datasets and Benchmarks): a multi-agent simulator, calibrated to resemble real transactions, that produces labelled synthetic datasets. **HI** and **LI** are the high- and low-illicit-rate "Small" variants. Because the data are synthetic the ground truth is complete.

**Files.** A transactions file (timestamp, source and destination bank and account, amounts and currencies, a laundering flag) and a **patterns file** listing each laundering *attempt* (a numbered block with a pattern type, e.g. CYCLE) and the transactions in it. Our loader (`data/ibm_aml.py`) joins pattern rows to transactions on (timestamp, accounts, amount); duplicate keys are paired deterministically, laundering rows first.

**Pattern types seen in positives:** CYCLE, GATHER-SCATTER, SCATTER-GATHER and STACK. The patterns file also contains BIPARTITE, FAN-IN, FAN-OUT and RANDOM attempts, which appear in the audit (section 8.2) but produce no positive candidates at tau 0.05 and max_len 6.

**Cleaning decisions.**
* **Self-loops dropped.** Rows with source equal to destination are removed and counted by the loader (HI-Small: 591,212 of 5,078,345 rows, 11.6%); node in/out totals therefore use inter-account flow only.
* **Currencies.** `Amount Paid` is used as is, in its own currency (15 currencies, no exchange rates in the data); volume ratios mix currencies. A stated limitation.
* **Unattributed laundering rows.** 1,968 of the 5,177 laundering rows belong to no attempt in the patterns file. A candidate whose laundering-flagged edges rest only on such rows is labelled **ambiguous**, and results are reported with ambiguous candidates excluded (primary) and counted positive.
* **What the pack records.** The pack's setup table gives the transactions kept, the windows and the candidate counts (section 8.3).

**Why the IBM data is not market wash trading.** IBM laundering patterns are often one-directional, may decay in amount and need not be regular in time, so the wash-trading behaviour scores need not describe them. The v3 proposal therefore uses IBM to validate the *pipeline machinery* and runs a pattern-behaviour audit first (section 8.2).

### 5.3 NFT data (planned, inspected, not run)

**Plan.** Ethereum NFT trades from Dune or Google BigQuery, with proxy wash-trade labels from Dune's `nft.wash_trades`, and a label-provenance audit. **What was done.** BigQuery access was confirmed and the table schemas were inspected. **Why it was not run.** Running queries on Dune is blocked on the free tier. **Leakage design.** `nft.wash_trades` combines five filters; filters 2 and 3 are cycle-type rules, so labels from them would leak the very structure our method detects. The planned leakage-controlled subset was therefore filters 4 and 5 only. No NFT result exists, so **we make no claim beyond synthetic and IBM data**.
