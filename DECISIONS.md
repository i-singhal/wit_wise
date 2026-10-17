# Decision Log

Running record of what we decided, why, and what evidence backed it. Newest phase at the bottom.

## Project concept
Split each squad's 100 reward points by judging every player against their own **discovered role** (soft GMM membership),
benchmarked on top-10 teams, weighted by how much each role moves team placement. Replaces the single shared formula in the
`Math` branch (`engine.py`), which used fixed pillar weights (45/35/20) for everyone.

## Phase 0 — Setup
| # | Decision | Why |
|---|---|---|
| D0.1 | Merged `origin/Math` into `main` (fast-forward) | Reuse its idea of squad-sum-normalised 100-point split, anti-gaming caps, kills-only baseline, Streamlit-style demo squad |
| D0.2 | Local `.venv` with pandas/sklearn/matplotlib/seaborn/plotly/pyarrow | System Python had no sklearn |
| D0.3 | `data/` and `.venv/` git-ignored | 660 MB CSV; reproducible from `prepare_data.py` |
| D0.4 | Final dashboard is a **single static HTML file** | Requested by user; no server needed to demo |

## Phase 1 — Data & EDA
Scripts: `prepare_data.py` -> `data/clean.parquet`, `eda.py` -> `eda/*.png`, `eda/findings.json`.

### Cleaning decisions
| # | Decision | Why / evidence |
|---|---|---|
| D1.1 | Dropped 1 row with NaN `winPlacePerc` | Known bad row in the Kaggle set |
| D1.2 | Kept only `squad`, `squad-fpp`, `normal-squad`, `normal-squad-fpp` (2.40M of 4.45M rows) | The 100-point split only makes sense inside a team |
| D1.3 | Removed suspected cheaters/bugs: kills with zero movement (1,036), longestKill>1000 (13), weapons>80 (9), heals>40 (46), kills>25 (89), teamKills>3 (91); 1,267 rows in union | Standard thresholds for this dataset; would otherwise sit at the top of every percentile |
| D1.4 | Kept only teams of **2–4 players** (1.62M rows, 500,845 teams, 25,565 matches) | 700k squad-mode rows sit in "groups" of 5+ (merged/custom lobbies); size-1 teams have nobody to split with. Group size is not 4 for ~30% of raw rows, so we do NOT assume 4 |
| D1.5 | Added `totalDistance = walk + ride + swim` | Math-branch used walk+ride; swim is 92% zeros, kept for completeness |
| D1.6 | **No `timeSurvived` in the data**; use `winPlacePerc` and `matchDuration` | Math-branch `engine.py` expects `timeSurvived`. Must be replaced by a placement-based survival proxy in Phase 2/3 |
| D1.7 | "Top-10 team" defined via `round(1 + (1-winPlacePerc)*(numGroups-1)) <= 10` using raw `numGroups` | First attempt used filtered team count and flagged 57% of players; fixed. Correct share is 44% because lobbies have ~25 teams |

### EDA findings that shaped the design
| # | Finding | Evidence | Implication |
|---|---|---|---|
| E1 | Half the players (51.6%) have **zero kills** | `03`, findings.json | Kills-only baseline gives them 0 points |
| E2 | Of zero-kill players, **23%** revived or assisted and **16%** dealt >100 damage | `10` | Strong case that non-killers contribute and are invisible today |
| E3 | 82% of teams contain at least one zero-kill player | `09` | The unfairness is the norm, not an edge case |
| E4 | Within-squad inequality (Gini) is high for kills (0.44), damage (0.42), heals (0.39), revives (0.36) but low for walk distance (0.16) | `12` | Squads are specialised on kills/support, but move together, so distance is a weak role signal |
| E5 | Strongest placement correlates: walkDistance 0.79, boosts 0.60, weaponsAcquired 0.51; kills only 0.37 | `06` | Kills are not the main driver of winning |
| E6 | Top-10 teams out-stat the rest by 2–5x on almost everything (kills 2.8x, revives 2.7x, boosts 4.8x) | `13` | Benchmark role stats are strongly survival-inflated |
| E7 | Stats rise monotonically with placement decile; damage and kills are flattest, heals/boosts/walk steepest | `14` | **Survivorship bias is real**: top-10 benchmarks must be rate-normalised (per minute / per distance / within-match percentile), not raw |
| E8 | 48–79% zero inflation in revives, assists, heals, DBNOs, rideDistance | `03` | Need log/rank transforms before GMM; raw-scale GMM would model the zeros |
| E9 | Modes differ in kill/damage/duration levels | `08`, `16` | Normalise within match (or at least within matchType) |

### Risks logged for later phases
- GMM on zero-inflated stats -> use within-match percentile ranks, then check cluster interpretability.
- Role-importance can mechanically favour fraggers (kills reduce enemies left) -> shrink/cap weights (see mentor plan, point 3).
- No stable player IDs -> roles are per-match behaviour, not identity.
- Fit roles on one set of matches, estimate benchmarks/importance on a held-out set.

### Phase 1 deliverables
18 charts in `eda/` (composition, target, distributions, correlation, stat-vs-placement, hexbins, duration, kill concentration,
zero-kill contributors, team stats vs placement, Gini, top-10 vs rest, survivorship, role-space hint, mode comparison,
team size, cleaning funnel).

## Phase 2 — Role discovery
Script: `roles.py` -> `roles/*.png`, `roles/gmm.joblib`, `data/roles.parquet` (adds `m0..m7`, `role_name`, `role_conf`, `split`).

| # | Decision | Why / evidence |
|---|---|---|
| D2.1 | Matches split 50/50 into **A (fit roles)** and **B (held out for benchmarks, importance, validation)** by `matchId`, seed 42 | Mentor point 4 of the plan: avoid circularity between role discovery and placement-based steps. Role shares are identical in A and B (e.g. Flex 42.0% vs 42.1%) |
| D2.2 | First attempt: GMM on raw within-match percentile ranks (k=6, ARI 0.999) **rejected** | Clusters were just an engagement ladder (low->high on everything). That is the survivorship factor from E7, not playstyle |
| D2.3 | **Features = within-match percentile minus teammates' mean percentile** (`d_*`), 9 stats: kills, damage, DBNOs, assists, revives, heals, boosts, walk, ride | The split happens inside a squad, so a role should mean "what this player did *more than their teammates*". Removes shared squad survival; exposes specialisation (E4) |
| D2.4 | Percentile (average ties) instead of raw/log values | Stats are 48–79% zeros (E8) and lobby scale differs by mode/duration (E9) |
| D2.5 | `GaussianMixture(full covariance, reg_covar=0.002, n_init=5)` | Full covariance captures correlated stats. A larger `reg_covar=0.01` was needed for the uncentered attempt; 0.002 suits the centred scale. First un-regularised run produced non-monotone BIC (tie-spike components) |
| D2.6 | **k = 8** | Chosen on stability + interpretability, not BIC alone (BIC/LL keep creeping up). Seed-ARI 0.993 at k=8 (0.64–0.80 at k=5–7, 0.94 at k=9, 0.75 at k=10); held-out LL plateaus at 9. Every k=8 cluster is distinct and nameable. k=9 only splits Wingman into two |
| D2.7 | Role names set from centroids (see table) | Roles are *labels for data-driven clusters*, so names can be changed without refitting |
| D2.8 | Soft membership kept (`m0..m7`); 19.9% of players have max membership < 0.6 | These are the hybrids the GMM was chosen for; hard label stored only for charts |

### The 8 roles (share of players)
| Role | Share | Signature (percentile advantage vs teammates) |
|---|---|---|
| Flex | 42% | ≈0 on everything (no standout trait) |
| Slayer | 15% | +kills, +damage, +DBNOs |
| Medic | 14% | +0.31 revives |
| Wingman | 9% | +0.29 assists, fewer kills |
| Survivor | 7% | well below teammates on kills/damage/DBNOs, level on heals/walk |
| Passenger | 6% | below teammates on everything |
| Apex | 4% | above teammates on everything (kills 3.5/match, boosts 3.3) |
| Rotator | 4% | +heals, +boosts, +ride, fewer revives |

### Findings & caveats
- **C2.1 Flex is a catch-all, and weak.** Flex players average winPlacePerc 0.41 vs ~0.67 for every other role. The most common squad is (Flex, Flex, Flex): when a whole squad dies early nobody stands out, so team-centring puts them all at ≈0. *Flex mixes "genuinely balanced" with "nothing happened".* Phase 3 must therefore score **absolute performance within role**, not just role membership.
- **C2.2 Passenger has the highest mean placement (0.70).** Centred roles are relative: a Passenger on a winning squad did less than carries, not nothing. Same reasoning: points come from the benchmark percentile, not from the role label.
- **C2.3 Squads with more distinct roles place better** (1 role: 0.26, 2: 0.51, 3: 0.65, 4: 0.70). Correlation only: winning squads have more to specialise in. Not claimed as causal; do not use as a feature.
- **C2.4 Roles are per-match behaviour**, not player identity (no stable player IDs).
- Raw-unit check (`roles/07`): Medic 1.3 revives/match, Wingman 1.4 assists, Slayer 2.4 kills, Apex 3.5 kills + 400 damage: roles are interpretable in real stats, not only in percentile space.

### Phase 2 deliverables
9 charts in `roles/`: k-selection (LL/ARI/BIC), centroid heatmap, radars, role sizes + hybrid-ness, role mix by mode/team size, PCA map, raw stats by role, squad compositions vs placement, role co-occurrence.

## Phase 3 — Benchmarks, scoring, 100-point split
Script: `score.py` -> `score/*.png`, `score/params.json`, `score/headline.json`, `data/scored.parquet`.

| # | Decision | Why / evidence |
|---|---|---|
| D3.1 | Scoring uses **absolute** within-match percentiles (`q_*`), unlike role discovery which used teammate-centred ones | Roles say *how* someone plays; the score must say *how well*, otherwise a Passenger on a winning squad looks "bad" (C2.2) and a Flex on a dead squad looks "average" (C2.1) |
| D3.2 | Benchmark per role = distribution of each stat among players of that role on **top-10 teams in split B** | Mentor plan step 2. B is held out from role fitting (D2.1). Benchmarks have 17k–85k players per role |
| D3.3 | Within-role score = Σ_f rubric_weight(role,f) × ECDF_role,f(player stat). **Mid-rank ECDF** | First version used right-sided ECDF; zero-ties inflated everyone (99.7% of zero-kill players got >10 pts). Mid-rank fixes ties |
| D3.4 | Rubric weight(role,f) ∝ max(centroid_f, 0) + 0.03, normalised | Stats that define a role count most; floor keeps every stat in play so Passenger/Survivor (negative centroids) are still judged on something |
| D3.5 | Player performance = Σ_role membership × within-role score (soft membership) | Hybrids blend rubrics instead of being forced into one (D2.8) |
| D3.6 | **Role importance** = ridge of team placement on soft role counts (split B, 4-player squads, controlling lobby size), z-scored, mapped to `1 + 0.15·tanh(z)` → range [0.86, 1.13] | Mentor plan step 4 plus the risk we logged: raw coefficients are confounded (Apex/Slayer co-occur with winning; Flex with early death). Capping at ±15% keeps a medic from being outscored by an average slayer by construction |
| D3.7 | **Contrast exponent γ = 3** applied before normalising to 100 (exposed as a dashboard slider) | At γ=1 the top earner of a 4-squad got 32 pts (≈ equal split, uninformative). γ=3 → 46 pts, between kills-only (59) and fixed pillars (43). `score/07_gamma_sensitivity.png` |
| D3.8 | Points normalised within squad to sum to 100; squad size 2–4 | Same guarantee as the Math-branch engine |
| D3.9 | Baselines: (a) kills-only; (b) Math-branch fixed pillars (45/35/20), with total distance as the survival proxy because no `timeSurvived` exists (D1.6) | Gives two reference allocations to compare against |

| D3.10 | **Two-lens score**: `raw = ((1−β)·style_score + β·absolute_score)^γ`, β = 0.4. The absolute lens is the Math-branch pillar formula (45% combat / 35% support / 20% tactical) applied to within-lobby percentiles | **Flaw found in the dashboard squad explorer**: a Passenger with 0 kills / 30 dmg / 1 revive got 69 pts while two Apex players with 3–4 kills and 300+ dmg got 13 each. Pure within-role percentiles let "best at a low-impact style" beat "average at a high-impact style". β sweep on held-out squads (γ=3): β=0 Passenger/Survivor ≈0.84/0.81× equal share, Apex 1.40×; β=0.4 → 0.62/0.59× and 1.58×; zero-kill players still average ~19 pts in 4-squads with 79% above 10. β=0.4 keeps style-relative fairness (60%) but anchors to real contribution. **β is a policy knob** |

### Results (split B)
- Zero-kill players (all squad sizes): mean points kills-only **12.8**, fixed pillars 24.2, ours **24.1**; share with >10 pts: 39% → **85%**.
- Top earner in a 4-player squad: kills-only 59.5, fixed 42.7, ours **43.9** (after D3.10).
- Importance multipliers: Apex 1.13, Slayer 1.10, Rotator 1.07, Survivor 1.04, Medic 1.00, Passenger 0.98, Wingman 0.88, Flex 0.86. Importance moves points by ~1.7 pts on average after D3.10 (3.1 before) (so it matters, but cannot dominate).
- **Caveat C3.1** Wingman and Flex multipliers are below 1 largely because those roles are over-represented in squads that die early; they are *not* evidence that assists are worthless. This is the main limit of learned importance on observational data, and the reason for the cap.
- **Caveat C3.2** γ is a policy knob, not a fitted parameter. Report results at γ ∈ {2,3,4}.

## Phase 4 — Team dynamics & behavioural patterns
Script: `dynamics.py` -> `dynamics/*.png`, `behavior/*.png`, `dynamics/summary.json`.

| # | Finding | Evidence |
|---|---|---|
| T1 | Six squad archetypes by role mix (k-means on soft role shares): Flex+Rotator squads place 0.28 and reach top-10 only 10% of the time; Passenger+Apex 0.75 / 73% top-10 | `dynamics/01` |
| T2 | Strongest synergies: Survivor+Flex (+0.116), Slayer+Apex (+0.111). Weakest: Flex+Wingman (−0.093), Flex+Medic (−0.054), Wingman+Slayer (−0.054) | `dynamics/02`. Ridge, confounded by survival; treat as associations |
| T3 | **Cohesion matters**: squads that move together place better (r = +0.39). Carry-dependence is mildly negative (top-earner share r = −0.10, point-Gini r = −0.08) | `dynamics/04`, `05` |
| T4 | Our #1 earner is **not** the top killer in 48% of 4-player squads | `dynamics/06` |
| T5 | Role coverage lift: Apex +0.22, Medic +0.20, Passenger +0.20, Survivor +0.16; Flex −0.28 | `dynamics/03` (raw, survival-confounded) |
| B1 | **External validation of roles**: behaviours not used for clustering line up with role labels. Wingman damage/kill = 162 vs 117–133 elsewhere (softens targets for teammates); Rotator swims 10.4 and has most teamKills/vehicle destroys among non-fraggers; Slayer/Apex have the longest kills (63/81 m) and streaks | `behavior/01` |
| B2 | Outcome quadrants: Winning fragger 31%, Early ghost 30%, Quiet survivor 27%, Glass cannon 12% | `behavior/02` |
| B3 | Hybrids (20% of players) mostly sit between Flex and a specialist role: Flex+Slayer 24%, Flex+Survivor 13%, Slayer+Wingman 8%, Slayer+Survivor 6%, Apex+Slayer 5% | `behavior/05` |
| B4 | **No kills-vs-support trade-off inside squads**: the teammate with more kills also tends to have slightly more revives (+0.16) and more distance; kills vs assists ≈ 0. Specialisation is about *who does more*, not *who gives up what*. (An earlier draft of this log claimed a negative trade-off; the data say otherwise.) | `behavior/04` |
