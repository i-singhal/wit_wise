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
