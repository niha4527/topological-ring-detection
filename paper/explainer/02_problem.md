## 2. Problem statement

### 2.1 What we are trying to catch

**Wash trading** is trading with yourself through a front: a group of colluding accounts repeatedly send an asset (or money) to one another so that the last recipient hands it back to the first. Nothing economic happens, but the trade count and volume go up, which can mislead other participants about demand. **Layering and cycling in money laundering** has the same shape: value is passed through several accounts and returns near its origin, so its source is hard to trace. The IBM AML data we use contains such laundering patterns with labels (section 5.2).

The unit of interest is a **ring**: a small set of accounts a1, a2, ..., ak and a directed cycle of payments a1 to a2 to ... to ak to a1, repeated or approximately repeated.

### 2.2 Why single transactions do not reveal it

Each payment in a ring looks ordinary: normal amount, normal counterparties, normal time. A transaction-level anomaly detector scores each payment in isolation and sees nothing odd. What is odd is the *pattern of connections*: money circulates and nets to about zero for each member. That is a statement about a *graph* (accounts are nodes, payments are directed edges), and specifically about a loop in it.

### 2.3 The two gaps this project targets

These come from section 1 of the v3 proposal (`Topological_Ring_Detection_for_Market_Manipulation_v3.md`).

1. **Density-based graph detection can miss sparse rings.** The common graph approaches (community detection, centrality scores, dense-subgraph mining, graph neural networks) are built to find tightly clustered groups. A ring can be sparse: each member trades with only two or three others, yet the loop persists. Cycle-based methods (strongly connected components, cycle enumeration) do find loops, but their answer is binary (a cycle is flagged or not): no strength, no stability, no type.
2. **Fixed detection thresholds decay.** Manipulators change ring size, timing and the use of decoy trades. A flagging threshold tuned once on history degrades as tactics drift.

The v3 proposal also notes prior persistent-homology work on blockchain networks that flags *that the network's shape changed* without isolating individual rings or typing them (section 3).

### 2.4 Scope

| Pattern | Status in this project |
|---|---|
| Closed-cycle wash trading and laundering cycles | **In scope**, the load-bearing target |
| Pump-and-dump coordination (a temporal motif, not a closed cycle) | Optional stretch track; done only as a **synthetic, no-topology** exploratory run (section 8.14); no real-data result |
| Spoofing | **Out of scope**: it is an order-book phenomenon (orders placed and cancelled without executing); settled transaction data has no signal for it |

### 2.5 What "detect" means here

We do not try to label whole accounts. We (a) generate *candidate rings* exactly, then (b) decide for each candidate whether it is a manipulation ring (classification), optionally what type it is (clustering) and how large its next-window activity will be (regression, synthetic only), and (c) keep the flagging threshold current under drift. The central scientific question is narrower: *given the candidate and its direction and behaviour features, do topological features add predictive value?* The primary comparison is therefore **B+D+T versus B+D** (section 4.9).
