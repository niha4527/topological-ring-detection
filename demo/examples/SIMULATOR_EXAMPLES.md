# Simulator examples

Paste a case into the Simulator box (tab 2) and press Score. The blocks here show only the first 12 lines; the verdicts are for the full input, which is in `demo/examples/inputs/<name>.txt` (copy from that file).
Format per line: `hour from to amount`. Hour is 0 to 168 (one week). `#` starts a comment.

Verdicts below were produced by running each block through the same models the Simulator uses. Scores are the model's ring score (higher = more suspicious); the cutoff is each model's own tuned threshold.

## Should flag

### Wash ring, 3 accounts, every 4h

Model result: B+D: FLAGGED (score 2.474); B+D+T_local: FLAGGED (score 4.796)

```
0.00 A B 1006.95
0.10 B C 1005.28
0.20 C A 995.10
4.00 A B 998.99
4.10 B C 1003.03
4.20 C A 1005.77
8.00 A B 990.57
8.10 B C 1006.72
8.20 C A 998.66
12.00 A B 990.04
12.10 B C 998.91
12.20 C A 1004.43
# ... (continues; use the file in demo/examples/inputs/ for the full text)
```

### Wash ring, 4 accounts, every 6h

Model result: B+D: FLAGGED (score 5.781); B+D+T_local: FLAGGED (score 8.960)

```
0.00 A B 1006.95
0.20 B C 1005.28
0.40 C D 995.10
0.60 D A 999.91
6.00 A B 1003.03
6.20 B C 1005.77
6.40 C D 991.88
6.60 D A 990.57
12.00 A B 998.66
12.20 B C 1005.25
12.40 C D 990.04
12.60 D A 998.91
# ... (continues; use the file in demo/examples/inputs/ for the full text)
```

### Wash ring, 5 accounts, every 6h

Model result: B+D: FLAGGED (score 7.794); B+D+T_local: FLAGGED (score 10.495)

```
0.00 A B 1006.95
0.20 B C 1005.28
0.40 C D 995.10
0.60 D E 999.91
0.80 E A 998.99
6.00 A B 1005.77
6.20 B C 991.88
6.40 C D 990.57
6.60 D E 1006.72
6.80 E A 998.66
12.00 A B 990.04
12.20 B C 998.91
# ... (continues; use the file in demo/examples/inputs/ for the full text)
```

### Ping-pong pair, every 3h

Model result: B+D: FLAGGED (score 2.292); B+D+T_local: FLAGGED (score 9.532)

```
0.00 A B 1006.95
0.10 B A 1005.28
3.00 A B 999.91
3.10 B A 998.99
6.00 A B 1005.77
6.10 B A 991.88
9.00 A B 1006.72
9.10 B A 998.66
12.00 A B 990.04
12.10 B A 998.91
15.00 A B 994.58
15.10 B A 1008.91
# ... (continues; use the file in demo/examples/inputs/ for the full text)
```

### Fast 3-ring, every 2h, large amounts

Model result: B+D: FLAGGED (score 10.199); B+D+T_local: FLAGGED (score 12.289)

```
0.00 A B 50347.43
0.10 B C 50263.77
0.20 C A 49755.07
2.00 A B 49949.49
2.10 B C 50151.59
2.20 C A 50288.72
4.00 A B 49528.35
4.10 B C 50335.77
4.20 C A 49932.77
6.00 A B 49502.11
6.10 B C 49945.39
6.20 C A 50221.54
# ... (continues; use the file in demo/examples/inputs/ for the full text)
```

### Ring with decoy trades to outsiders

Model result: B+D: FLAGGED (score 3.550); B+D+T_local: FLAGGED (score 6.524)

```
0.00 A B 1000.88
0.05 A Z 300.27
0.10 B C 1008.32
0.15 B W 302.50
0.20 C A 1002.51
0.25 C X 300.75
4.00 A B 1008.18
4.05 A W 302.45
4.10 B C 995.19
4.15 B Y 298.56
4.20 C A 993.83
4.25 C W 298.15
# ... (continues; use the file in demo/examples/inputs/ for the full text)
```

### Two rings in one input

Model result: B+D: FLAGGED (score 7.058); B+D+T_local: FLAGGED (score 10.208)

```
0.00 A B 1006.95
0.00 P Q 2517.37
0.10 B C 1005.28
0.20 C A 995.10
0.20 Q R 2513.19
0.40 R S 2487.75
0.60 S P 2499.77
4.00 A B 998.99
4.10 B C 1003.03
4.20 C A 1005.77
6.00 P Q 2507.58
6.20 Q R 2514.44
# ... (continues; use the file in demo/examples/inputs/ for the full text)
```

## Should not flag

### One-way chain, no loop

Model result: B+D: not flagged (no candidate ring: no directed cycle of length 2-6 survives the sparsification cutoff among these accounts); B+D+T_local: not flagged (no candidate ring: no directed cycle of length 2-6 survives the sparsification cutoff among these accounts)

```
0.00 A B 1006.95
0.10 B C 1005.28
0.20 C D 995.10
4.00 A B 1003.03
4.10 B C 1005.77
4.20 C D 991.88
8.00 A B 998.66
8.10 B C 1005.25
8.20 C D 990.04
12.00 A B 994.58
12.10 B C 1008.91
12.20 C D 1008.03
# ... (continues; use the file in demo/examples/inputs/ for the full text)
```

### One-off triangle, 3 trades

Model result: B+D: not flagged (score -22.352); B+D+T_local: not flagged (score -25.694)

```
10 A B 500
11 B C 500
12 C A 500
```

### Irregular loop, random timing and sizes

Model result: B+D: not flagged (score -0.206); B+D+T_local: not flagged (score 1.974)

```
1.30 P Q 860.34
1.40 Q R 1060.37
1.50 R P 828.97
8.14 P Q 946.28
8.24 Q R 823.20
8.34 R P 1002.97
12.15 P Q 973.46
12.25 Q R 827.94
12.35 R P 836.29
19.70 P Q 1130.74
19.80 Q R 849.52
19.90 R P 889.30
# ... (continues; use the file in demo/examples/inputs/ for the full text)
```

### Single payment

Model result: B+D: not flagged (no candidate ring: no directed cycle of length 2-6 survives the sparsification cutoff among these accounts); B+D+T_local: not flagged (no candidate ring: no directed cycle of length 2-6 survives the sparsification cutoff among these accounts)

```
5 A B 100
```

### Two-way trade, one round each

Model result: B+D: not flagged (score -22.496); B+D+T_local: not flagged (score -17.398)

```
20 A B 1000
90 B A 1000
```

### Star: A pays four accounts, no returns

Model result: B+D: not flagged (no candidate ring: no directed cycle of length 2-6 survives the sparsification cutoff among these accounts); B+D+T_local: not flagged (no candidate ring: no directed cycle of length 2-6 survives the sparsification cutoff among these accounts)

```
1 A B 300
11 A C 300
21 A D 300
31 A E 300
```

## Edge case

These look like ordinary behaviour to a person but the model flags them. They are limits of the model, not bugs in the Simulator: repeated, near-equal round trips between the same accounts are exactly what it learned to call wash trading.

### Hub that pays and is repaid by 4 accounts

Model result: B+D: FLAGGED (score 7.621); B+D+T_local: FLAGGED (score 8.719)

```
0.00 A B 400.00
0.10 A C 400.00
0.20 A D 400.00
0.30 A E 400.00
1.00 B A 395.00
1.10 C A 395.00
1.20 D A 395.00
1.30 E A 395.00
4.00 A B 400.00
4.10 A C 400.00
4.20 A D 400.00
4.30 A E 400.00
# ... (continues; use the file in demo/examples/inputs/ for the full text)
```

