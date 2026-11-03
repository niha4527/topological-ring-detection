# Speaking script, in depth: slides 8 to 11

How to read this:
- Plain text under "SAY" is what you speak.
- [KNOW] blocks are background for you. You don't need to say them, but you should be able to answer questions about them.
- Paragraphs marked (optional) can be cut if you are short on time.
- Full version is about 3 minutes per slide, about 12 minutes for the four. The cutting guide at the end gets it to 6.

Before presenting, fix in the deck: slides 8 and 15 say 210 tests and the repo now has 220. Slide 11 should say the learned weights
come from synthetic data.

---

# Slide 8: System architecture

## SAY

"This slide shows how data flows through the system. There are eight stages. Each is its own package in `src/ringtda/`, so each one can be
tested alone. I'll go through them in order and say what goes in and what comes out.

**Stage 1, data ingestion.** We have two sources, IBM's AML data and our own synthetic generator. Both are turned into the same simple
table: a timestamp, a sender account, a receiver account, and an amount.

**Stage 2, graph construction.** This is where most of the design decisions are. We cut time into windows. On IBM data a window is 5 days
and moves forward 1 day at a time. We use 5 days because laundering cycles in the IBM data can last up to 96 hours, and a shorter window
would cut them in half.

Inside one window, all payments from account A to account B are merged into one directed edge. The edge carries the total amount, the
number of payments, and how regular the timing is.

Then every edge gets three suspicion scores between 0 and 1, where higher means more suspicious. The first is reciprocity: do A and B pay
each other similar amounts. The second is flow balance: does each account send out about as much as it receives. That is what happens to
every member of a ring, because the money comes back around. The third is timing regularity: is the cadence clockwork, like a bot. It's one
divided by one plus the variation in the gaps between payments.

Each score is turned into its rank inside the window, so no single score dominates because of its scale. The edge weight is one minus the
average of the three ranked scores. So **a lower weight means a more suspicious edge**.

For example, a ring edge with ranked scores of 0.0, 1.0 and 0.987 gets weight one minus 1.987 over 3, which is 0.338.

**Stage 3, candidate rings.** We keep only the most suspicious edges. On IBM data that is the lowest-weight 5 percent. Then we find strongly
connected components, groups of accounts where you can get from any one to any other by following payments. Every loop lives inside one.
Inside each component we list every directed loop of two to six accounts. Each loop is a candidate ring. A ring's **birth** is its largest
edge weight, which is the moment the loop first exists as the filter loosens.

We list loops exactly, rather than detecting them from topology, because the persistence library tells us a loop exists but not which
accounts form it. We need the exact accounts, edges and transactions.

**Stage 4, persistence.** For each candidate we take the ring and its 1-hop neighbours and build a directed flag complex. It adds each
account at level 0, each directed edge at its weight, and a triangle for every ordered triple A, B, C where A→B, A→C and B→C all exist. A
circular triangle A→B→C→A never gets filled, so its loop stays open forever. A one-way triangle is filled immediately. If a fourth account
sends to all three members, the loop is filled later, when those edges enter. The library returns a bar for each loop: when it was born and
when it died. We turn those bars into numbers.

**Stages 5 to 8.** Stage 5 computes the features, three groups called B, D and T. Stage 6 trains the models. Stage 7 is the adaptive
threshold. Stage 8 is evaluation, with labels, grouped cross-validation and bootstrap intervals.

The highlighted stages, 3 and 4, are the topology core. One rule runs through the whole design: the logic lives in the package, notebooks
only draw plots, and 220 automated tests check that each stage is correct."

## [KNOW]

- **Why rank-normalise.** Raw reciprocity is 0 for most edges (most pairs don't pay each other back), so the scores have different scales and shapes.
  Ranks put them on one scale. Ties take the lowest rank, so the big block of zero-reciprocity edges maps to 0 and not to the middle.
- **Proposition 1.** In a ring of three or more accounts nobody pays back directly, so reciprocity is 0 on every ring edge. With equal weights,
  the weight can't go below 1/3, however balanced and regular the ring is. This is why we later learn weights.
- **Persistence vocabulary.** A *filtration* is the process of letting edges in from most to least suspicious. *Persistence* is how long a loop
  survives while that happens. An H1 bar is one loop. "Dies" means something fills it.
- **Two persistence backends.** The default is flagser (fast, float32). We also wrote a pure-Python reference version, and a test requires them
  to agree. That's how we know the topology numbers are right.
- **Cap on cycle listing.** At most 20,000 cycles per component, and any component that hits the cap is recorded, never silently dropped.
- **De-duplication does nothing.** The design said to remove near-duplicate rings sharing 80% of edges. We found that a k-cycle shares at most
  k−2 edges with any other loop up to its length, so nothing is removed up to length 9. So near-copies all stay as candidates. This is why we
  use grouped cross-validation later.

(optional) "One result of building it carefully: on IBM HI-Small, the first window has 2.2 million transactions after cleaning and produces 1,113
candidate rings in under a second of enumeration. So the candidate step is cheap. The expensive step is the per-ring features."

---

# Slide 9: Project workflow

## SAY

"This slide is the order in which we built things, and what each phase was meant to prove. We tested each phase before building on the next.

**Phase 1, the synthetic generator.** We built a fake world: random background traffic among 400 accounts over a week, with planted
structures we know the answer for. Wash rings of 2, 3 and 5 accounts. Rings inside busy neighbourhoods. Rings with internal chords. One-way
triangles that look like rings but aren't. Triangles filled by a fourth account. And legitimate look-alikes: arbitrage loops, market makers
and exchange hot wallets. It was meant to give us exact ground truth, so every later stage can be tested against known answers.

**Phase 2, rings and persistence.** We built candidate-ring generation and the persistence code. These ended in hard pass-or-fail tests:
every planted ring is a candidate; a cyclic triangle has a persistent loop; a transitive triangle has none; a triangle filled by a fourth
account dies. We also learned something here. With default settings only 7 of 24 planted dense rings keep an open loop, because their
neighbours fill them in.

**Phase 3, the IBM AML study.** We loaded IBM's data and built a labelling rule, features and grouped cross-validation. Before using our scores
on IBM we audited them, because those scores were designed for wash trading and laundering behaves differently. Then we ran the main
experiment: classifiers on B, then B plus D, then B plus D plus T. We also ran the other way round: train on HI, test on LI, and the reverse.

**Phase 4, typing and severity.** We clustered positive rings to see if they group by laundering pattern type, and we ran a severity regression
on synthetic data. Both are exploratory.

**Phase 5, NFT real data.** We planned a real-data test on NFT wash trades. We cut it, because the free Dune tier doesn't allow running the
queries we needed.

**Phase 6, adaptive threshold.** We tested several policies for updating the alarm threshold as tactics drift, in a simulation.

**Phase 7, pump-and-dump.** A small optional track on synthetic data. It uses trade timing and no topology.

The last box is the write-up, which is in progress. Along the bottom is the path every window follows: transactions, graph, edge weights,
candidate rings, features, classifier score, threshold, and the flagged rings."

## [KNOW]

- **Two kinds of checkpoint.** *Engineering gates* are hard tests of correctness, such as "planted rings are recovered". If one fails, it is a bug.
  *Reporting checkpoints* are scientific results, reported with confidence intervals whatever they are. We never turn a performance number into a
  test that must pass, because that invites tuning until it passes.
- **Why the time-ordered split failed.** We tried "train on early rings, test on later ones". HI-Small left zero test candidates and LI-Small left one,
  because rings recur so almost every window mixes early and late groups. We dropped it and reported it as unusable. Transfer between HI and LI became
  the out-of-sample check instead.
- **The audit result.** Edge-level AUC of CYCLE pairs against background: 0.794 for the combined weight, 0.795 for balance, 0.601 for timing. So our weights
  work on IBM cycles mostly through balance. That's why we call IBM a test of the pipeline, not of the wash-trading weights.

(optional) "Along the way we caught several of our own errors. The first classification run produced a confidence interval that contradicted its
own summary. The cause was a division by zero at the top of the ranking that dropped some bootstrap draws and biased the intervals upward. We fixed
it and reran, and the intervals got wider. We also found that a topology feature that is constant within a window was just identifying the time
window, so we separated it out. Each time a result looked better than it should, we looked for an artefact before believing it."

---

# Slide 10: Dataset

## SAY

"We use two kinds of data.

**First, IBM's AMLworld.** It's synthetic: IBM generated it with a simulator, but it comes with complete labels. For each laundering transaction we
know which attempt it belongs to and which pattern type, such as cycle, fan-out, fan-in, or random. We use the Small versions. HI means a high
proportion of illicit transactions, LI a low one.

The raw table has timestamps, banks, accounts, amounts and currencies, and a laundering flag. Transactions have no ID, so we join the patterns file
to them on timestamp, banks, accounts and amount. We drop transfers from an account to itself, 11.6 percent of HI-Small. We use amounts in their own
currency, so amounts across currencies are mixed, and that's a limitation.

On the slide: HI-Small has 4,487,133 transactions after cleaning, in 13 windows, giving 8,074 candidate rings. LI-Small has 6,119,572 transactions
in 12 windows and 10,206 candidates.

**How we label a ring.** A candidate is positive if at least 80 percent of its edges are backed by transactions from one laundering attempt. It's
negative if none of its edges are. Anything else is ambiguous. On HI-Small that gives 131 positive rings, 519 ambiguous and 7,424 negative. LI-Small has
38 positive, 486 ambiguous. The main results leave out the ambiguous rings, and a second variant counts them as positive. The conclusion doesn't change.

**The number that matters most is positive groups: 14 and 10.** Rings recur across overlapping windows, and many positive rings are near-copies. The 131 positive rings on
HI-Small come from only 43 laundering attempts, and merge into 14 groups. A group is a set of rings that share edges, so they can't be treated as independent.
So our real sample size is 14 and 10, not 131 and 38. That's why our confidence intervals are wide.

Also, positives are rare: 1.7 percent of HI-Small candidates and 0.39 percent of LI-Small. That's why we measure PR-AUC and compare it with that rate, not with 50 percent.

**Second, our own synthetic generator.** It plants three kinds of wash ring: isolated, in dense neighbourhoods, and with internal chords. It plants
topology-sensitive negatives: one-way triangles, and loops filled by a fourth account. And it plants hard negatives, which are legitimate loops that look
like rings: arbitrage bots, market makers, and exchange hot wallets. We model those on documented real behaviour so we don't invent something conveniently
separable. We also have knobs for evasion and drift: decoy trades, timing jitter and amount noise. We run six random seeds and four conditions, with and
without hard negatives, and with and without internal chords in dense rings. Every ring has exact ground truth."

## [KNOW]

- **Why 5-day windows with a 1-day stride.** CYCLE attempts in HI-Small last up to 96 hours. A shorter window would cut them. Overlapping windows mean the same ring
  appears in several windows, which is why we group.
- **Why IBM is "pipeline only".** Laundering isn't wash trading. Laundering patterns can be one-directional, decaying and irregular. So IBM tests our machinery, not
  our wash-trading weights.
- **Unattributed laundering.** In HI-Small, 1,968 of 5,177 laundering rows belong to no attempt in the patterns file. A candidate resting only on those rows is ambiguous.
- **Synthetic numbers.** Prevalence ranges from 0.0203 to 0.0481 across the four conditions, with 93 to 103 positives in 46 to 55 groups, so the synthetic data has more positive groups than
  IBM. The generator's designer decides how separable the classes are, so synthetic results describe the generator only.

(optional) "Why use two sources? IBM has labels independent of our features, but only a few positive groups. Our generator has many more, but we built it, so it
may be easier than reality. Neither is enough alone, so we report both and never mix their conclusions."

---

# Slide 11: Features, models and optimization

## SAY

"Each candidate ring is described by three groups of features.

**B is the graph baseline, 12 features.** It describes how connected the ring's accounts are: their degrees, their betweenness (how often they lie on
shortest paths between other accounts), their clustering coefficient (how much their neighbours are linked to each other), the variance of edge weights
around them, and the density and size of their neighbourhood. These are the kinds of features density-based detectors use. On IBM windows we estimate betweenness
from 100 sampled sources, because exact betweenness is too slow on windows with 400,000 accounts.

**D is direction and behaviour, 17 features.** The ring's length, its birth, its edge weights, the mean and minimum of each ranked score along the ring,
how unbalanced its members' in and out degrees are, and volume and timing statistics. This is what the ring actually does.

**T is topology, 311 features.** For each ring we take persistence landscapes of its neighbourhood, a fixed-size summary of the loops and how long they last,
plus counts of loops, total persistence, and persistence entropy. We compute the same for the whole window. A loop that never dies gets a death value equal to the window's
largest weight, so the numbers stay finite. We also run an exact test of whether the ring's own loop is ever filled by its neighbours.

**Ring birth is in D, not T.** It's just a function of the ring's own edge weights. If we counted it as topology, topology would get credit for something that needs
no homology at all. Our main comparison is B plus D plus T against B plus D. The only difference between the two is topology, so any change in performance is caused
by topology and nothing else.

**Models.** Logistic regression, a random forest of 300 trees, and a support vector machine with an RBF kernel. All have fixed settings and class weights that
give rare positives more influence. We never tune anything, because with 14 positive groups any tuning would overfit. The scaler is fitted only inside each training
fold, so nothing leaks.

**Evaluation.** Five-fold cross-validation, repeated five times, where a group is never split between training and test. We report PR-AUC with 95 percent
bootstrap intervals that resample whole groups, and we compare feature sets using the same resamples.

**Optimization.** We did three optimization problems.
One: logistic regression trained by gradient descent. The loss is convex, so a step of 1 over L, where L measures how sharply the loss curves, is guaranteed to
converge. With a step matched to the local curvature it reached a gap of one in a million in 322 iterations.
Two: learning the three edge-score weights. We used projected gradient descent on a simplex, which means the weights stay non-negative and add up to one. The learned
weights were about 0 for reciprocity, 0.21 for balance and 0.79 for timing regularity. That lifted ring-birth average precision from 0.57 to 0.9997. This was on our
synthetic data. The IBM classification uses equal weights.
Three: the alarm threshold. We update it online by gradient ascent on a reward of true positives minus lambda times false positives, with labels arriving two windows late.
On a drifting stream it kept about 97 percent of the best possible reward, against about 24 percent for a fixed threshold."

## [KNOW]

- **What "birth" means.** Birth is the filter level at which the loop's last edge enters, so the loop first exists. For a ring that equals its largest edge weight.
- **Landscapes.** A persistence diagram is a variable-size set of (birth, death) pairs, which a classifier can't use directly. A landscape turns it into a fixed-length vector.
  We use 3 landscapes of 50 grid points on [0, 1.1]. T has 156 per-ring columns plus 155 whole-window columns that are the same for every ring in a window.
- **`T_win` trap.** The 155 window columns are constant within a window. They scored 0.121 alone on HI-Small, which means they identify windows. That is why the demo model leaves them out.
- **Why PR-AUC.** It is the area under the precision-recall curve. Precision is the share of flagged rings that were real; recall is the share of real rings found. A random scorer gets the
  prevalence, so on HI-Small chance is 0.0173, and B alone scores about 0.08 to 0.09, which is 4 to 5 times chance.
- **The numbers for the two threshold results.** The optimization report gives 0.97 (gradient ascent), 0.86 (re-tune every 5 windows), 0.24 (fixed). The paper averages six stream seeds and gives 0.981,
  0.861, 0.298. They are different settings, so say which you mean if asked. The bandits lost to re-tuning.
- **Class weights.** "Balanced" weights each class inversely to its frequency, so the 1.7% positives count as much as the 98.3% negatives during training.

(optional) "A note on honesty here: topology gave no detectable gain on IBM, so the optimization and learned weights are an improvement to the synthetic pipeline, not to the IBM classifier. We say which is which."

---

# Cutting guide (to fit about 6 minutes)

- Drop every paragraph marked (optional).
- Slide 8: say stages 2 to 4 in full (about 90 seconds) and stages 1, 5 to 8 in one sentence.
- Slide 9: say Phases 1 to 3 in full, and Phases 4 to 7 in one sentence each.
- Slide 10: say the IBM numbers, the labelling rule, and the 14 positive groups point, then one sentence on the synthetic generator.
- Slide 11: say B, D, T, the ring-birth point and the models, and cut optimization to: "We also learned the edge weights and an adaptive threshold, covered on the results slides."

# Handover to the results slide

"So we have the data, the features and the models. Next, our results: what behaviour features do, and what topology adds."

# Likely questions on these slides

- **Why groups?** Near-copies of the same ring appear in many windows, so a random split would put copies in both training and test and inflate scores.
- **Why exclude ambiguous rings?** They only partly match a laundering attempt. A second variant counts them as positive, and the conclusion doesn't change.
- **Is the IBM data real?** No, it's synthetic with exact labels. It tests our pipeline, not real-world performance.
- **Why exact cycle listing instead of finding loops with topology?** The persistence library doesn't report which accounts form a loop, and we need the exact accounts.
- **Why not tune the models?** With 14 positive groups, tuning would overfit, and the result would be too optimistic.
- **Why is ring birth in D?** It's a plain function of the ring's edge weights, so counting it as topology would overstate topology's contribution.
- **What do the learned weights do on IBM?** Nothing. They were learned and tested on synthetic data. The IBM classification uses equal weights.
