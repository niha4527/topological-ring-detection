## 4. Our solution: the concept, from scratch

The idea in one paragraph: turn each window of transactions into a directed graph whose edge weights say how suspicious each account pair's behaviour is; list candidate rings exactly; and for each candidate ask a topological question, "is this loop ever *filled in* by other accounts' edges as we admit less-suspicious edges?", whose answer is a continuous, comparable set of features. The rest of this section builds each piece from a tiny example.

### 4.1 One graph per time window

Transactions have a timestamp, a sender, a receiver and an amount. We slice time into windows of a fixed length (5 days, sliding by 1 day on IBM; one-day windows in the severity study) and, inside each window, collapse all payments from a to b into **one directed edge a to b** carrying: the total volume, the number of payments, and the coefficient of variation (CV) of the gaps between payments (std of gaps divided by mean gap; small CV = regular timing). The graph is **directed** because a to b and b to a are different edges.

*Example.* In one window: a pays b ten times, b pays c ten times, c pays a ten times, all at regular intervals and for similar amounts. The window graph has three edges forming a loop. Another account d pays a once.

### 4.2 Suspiciousness weights, and why lower means more suspicious

Each edge gets three scores in [0, 1], higher = more suspicious (the formulas are in section 6.2):

* `s_rec` **pair reciprocity**: do a and b trade back and forth in balanced volumes?
* `s_bal` **node flow balance**: do both endpoints receive about as much as they send? (Money that circulates nets to about zero.)
* `s_reg` **timing regularity**: are payments evenly spaced? (Bots and scripted wash trades are regular.)

Each score is replaced by its *rank within the window* (so no score dominates by scale), the three ranks are averaged, and the edge weight is `1 minus that average`. So **a suspicious edge has a low weight**. We choose "low = suspicious" because persistent homology is built on a *sublevel-set filtration*: it admits cells in order of increasing value, so suspicious edges must have small values to enter first.

### 4.3 Filtration and sublevel sets

Imagine a threshold t that rises from 0 to 1. At each t, keep only edges with weight at most t. That nested family of graphs (more edges as t rises) is a **filtration**. At t = 0.1 only the very suspicious edges exist; at t = 1 everything does. A loop made of suspicious edges appears early (at a low t); a loop that needs a less suspicious edge appears later. The value of t at which a ring first exists is its **birth**: the largest weight among its edges (all its edges must be present). Ring birth is a natural suspiciousness score for the ring, and it is a feature of the ring's *own* edges, which is why it sits in D, not T (section 4.8).

### 4.4 Simplicial complexes, homology and persistence

A graph has only nodes (0-dimensional cells) and edges (1-dimensional). A **simplicial complex** adds higher cells: triangles (2-cells) and so on, glued along edges. The **flag complex** rule is: wherever a set of nodes is pairwise connected by edges, fill that whole clique with a cell.

**Homology** counts holes. **H0** counts connected components. **H1** counts independent loops that are not filled in by 2-cells: a hollow triangle has one H1 class (a loop with nothing inside); a filled triangle has none.

**Persistent homology** follows these holes along the filtration. A hole is **born** at the t where its loop first exists and **dies** at the t where something fills it. The pair (birth, death) is a **bar**; the set of bars is a **persistence diagram**; a long bar means a loop that lasts across many thresholds. A loop that is never filled has death = infinity: an **infinite bar**. Because machine-learning features need finite numbers, we **cap** infinite deaths at a documented value (by default the window's largest edge weight; section 6.5) and count infinite bars separately before capping.

### 4.5 Why a DIRECTED flag complex

This is the key design choice (v3 section 3.2). Take the three-edge loop a to b to c to a (**circular triangle**) and the three edges a to b, b to c, a to c (**transitive triangle**: one-way flow, no circulation).

```
 circular (a ring)              transitive (one-way flow)

     a                               a
    / ^                             / \
   v   \                           v   v
   b -> c                         b --> c
 a->b, b->c, c->a               a->b, b->c, a->c
```

* In an **undirected** clique complex both are three pairwise-connected nodes, so both are filled by a triangle: the ring's loop dies instantly and the two cases cannot be told apart.
* In a **directed flag complex** a triangle is filled only if the three edges can be ordered as a "source, middle, sink" (a to b, a to c, b to c). The transitive triangle has that ordering: it is filled, so it has no H1 class. The circular triangle has none: it is *never filled*, so its H1 class survives as an infinite bar. This is exactly "value circulates" versus "value flows one way".

The implementation (`topology/persistence.py`) builds cells exactly this way: nodes at filtration 0, directed edges at their weight, and 2-cells for ordered triples (a, b, c) with edges a to b, a to c and b to c, entering at the largest of their edge weights. A reciprocal pair a to b and b to a gives two distinct edge cells, so ping-pong trading forms its own loop.

### 4.6 What persistence adds beyond enumeration

*A third example: a ring filled by a fourth account.* Take the circular triangle a, b, c with weights 0.1 and add d with edges d to a, d to b, d to c at weight 0.6. The triples (d, a, b), (d, b, c), (d, c, a) are transitive, so as t reaches 0.6 the loop gets filled and the H1 class **dies at 0.6**. Enumeration still lists a, b, c as a ring; persistence says the loop is open only between 0.1 and 0.6 (a bar of length 0.5). If d had been absent the bar would never die.

So persistence carries *extra* information only when the ring's neighbourhood interacts with it: a fourth account fills it (shortcut-filled), chords split a long ring into shorter cycles (chorded), or many incidental loops crowd it (dense). For an **isolated** ring in a sparse neighbourhood the H1 class is born at the ring's birth value and never dies, so persistence is nearly redundant with enumeration. This is the central hypothesis of the project, and it is why synthetic scenarios (section 5.1) have these three shapes.

### 4.7 Why we find rings by SCC and cycle enumeration, not by homology

The libraries that compute directed flag persistence (flagser, via pyflagser) return the persistence pairs but **no representative cycle**: they tell us that a loop exists with some (birth, death) but not *which accounts* form it. For a detection system we need account-level answers. So localisation is done combinatorially and exactly (`topology/candidates.py`): (1) sparsify, keeping only edges with weight at most a cutoff tau; (2) find strongly connected components; (3) enumerate simple directed cycles up to length 6 inside each component (capped at 20000 cycles per component, with capped components reported, never silently dropped); (4) de-duplicate. Persistence then supplies *features* of each candidate's neighbourhood.

### 4.8 Persistence landscapes and the three feature families

A persistence diagram has a variable number of points, which classifiers cannot ingest. A **persistence landscape** (Bubenik, JMLR 2015) turns a diagram into a few piecewise-linear functions of the threshold; sampled on a fixed grid they give a fixed-length vector. We use 3 landscapes of 50 grid points on [0, 1.1] for the H1 diagram.

* **B (graph baseline)**, 12 columns: member degrees, betweenness, clustering, neighbourhood weight variance, density and size. The density/centrality baseline of section 3.
* **D (direction/behaviour)**, 17 columns: ring length, **ring birth**, edge weights, the mean and minimum of each rank-normalised score along the ring, member degree imbalance, volume and timing statistics.
* **T (topological)**, 311 columns: for the ring's neighbourhood (**T_local**, 156 columns) and for the whole sparsified window (**T_win**, 155 columns, identical for every ring in a window): landscape values, counts of H1 and H0 classes, infinite-bar count, total persistence, persistence entropy and, for T_local, `T_local_ring_inf`, an exact test that *this ring's own loop* is never filled.

**Why ring birth is in D, not T.** Birth is a function of the ring's own edge weights; counting it in T would credit topology for something enumeration already gives (the proposal's own warning). T contains only quantities that need a homology computation.

### 4.9 Why the primary comparison is B+D+T versus B+D

If we compared T against nothing, any gain could come from direction or behaviour information. B+D is the strongest non-topological feature set; adding T and asking whether performance rises isolates the *topological increment*. A null result there is reportable and is what we found on IBM (section 8.3). Secondary comparisons are B+D versus B (does direction/behaviour help?) and, labelled exploratory, B+D+T_local and B+D+T_win versus B+D (separating the local ring features from the window-constant ones).

### 4.10 The adaptive-threshold idea

A candidate is flagged when its score is at most a threshold tau (low ring birth = suspicious). The **reward** per window is `TP minus lambda times FP` (true positives minus a penalty for false positives). As tactics drift the best tau moves, but labels arrive late: you learn that a flagged ring was real only d windows later. The project compares:

| Policy | Idea |
|---|---|
| fixed | tune tau on the first few windows, freeze |
| re-tuned (every M windows) | refit the best tau on recent *labelled* windows, with the same lag |
| online gradient ascent | nudge tau uphill on a smoothed reward using the newest labelled window |
| discounted UCB | a bandit over K = 10 fixed thresholds that discounts old rewards |
| sliding-window Thompson sampling | a bandit that samples from per-arm beliefs built from the last W windows only |
| oracle | the best tau for each window using its own labels (an upper bound, not achievable) |

The proposal's primary comparison is *bandit versus re-tuned at the same lag*.
