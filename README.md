# Fair Share: a contribution ledger for PUBG squads

## The idea

Kill-share rewards the person who lands the last shot and ignores everyone who set it up. Fair Share splits each team's 100 points across five contribution pillars. Each pillar gets a budget learned from what actually moves a team's placement, and each player is scored against their own lobby. The result is a **receipt** per player: "34 points = 12 damage, 9 revives, 8 navigation, 5 floor." The panel can audit every point. We back the rule with a leave-one-out analysis ("what would this team's placement have been without this player?") and stress-test it against kills-only and equal-split baselines.

## What we are optimizing for

There is no per-player ground truth, so the supervised target is **team-level**, and the allocation is derived from it.

- **Target column:** `winPlacePerc`, collapsed to one row per team (`matchId` + `groupId`), since teammates share it. It is the only outcome label we have.
- **What we learn:** how much each behaviour is worth to team success. We don't care about the model's accuracy for its own sake. We care that it gives defensible pillar weights.
- **What we then optimize or tune:** the allocation rule's design choices, judged on:
  - **Validity:** it agrees with the leave-one-out contribution from the model.
  - **Style parity:** each archetype's average points track its marginal value.
  - **Stability:** ranks hold under weight perturbation.
  - **Gaming resistance.**
- **Leakage caveat:** `killPlace` and `walkDistance` partly *result from* surviving. We train with and without them and report both.

## Parameters

| Group | Details |
|---|---|
| Pillars | Offense (damage, kills, assists, DBNOs, unconverted damage), Support (revives, assists), Survival (heals, boosts), Navigation (walk distance, vehicle use), Resourcing (weapons acquired) |
| Scoring | Within-match percentile, with log or sqrt transforms and winsorising |
| Allocation knobs | Pillar weights, softmax temperature τ, participation floor (~5), per-player cap (~60), `teamKills` penalty |
| Scope | Standard solo, duo and squad modes only, allocated among observed members, with cheater and glitch flags excluded |

## What sets us apart

1. We credit **unconverted damage**, which is invisible in kills.
2. Support is **opportunity-adjusted**: revives are judged against how many teammates went down.
3. The allocation is **ratio-consistent**, so a missing teammate doesn't reshuffle everyone else.
4. We validate with **global sensitivity analysis** of the design choices, not a single weight set.
5. Every point is explained on a receipt.
6. We are honest about the no-ground-truth limit and define fairness through axioms plus tests.

## Defining fairness and success

**Axioms (testable):**
- **Efficiency:** points always sum to exactly 100.
- **Symmetry:** identical stats get identical points.
- **Monotonicity:** more of a good stat never lowers your points.
- **Null-player handling:** a member who did nothing gets only the floor.
- **Ratio consistency:** a missing teammate doesn't change the others' relative shares.

**Outcome metrics:**
- **Style parity:** each archetype's average share against its marginal contribution to team placement.
- **Non-killer share:** points going to players with zero kills, compared with the baseline.
- **Within-team Gini:** a spread that is neither flat nor winner-takes-all.
- **Validity:** agreement with a held-out team-outcome model.

**Baselines:** kill share, damage share, equal split (25 each).

## EDA: key insights (each becomes a dashboard chart)

1. **The kills-only baseline breaks.** 57% of players have zero kills, and some teams have no kills at all (the baseline divides by zero). Chart: share of points a zero-kill player would receive under each rule.
2. **Teammates differ a lot.** Within-team Gini for kills, damage and revives. Chart: distribution of that inequality by feature, showing where the 100 points have to differ.
3. **Survival and navigation drive placement.** Walk distance decile against placement (about 0.10 rising to 0.87), and boosts against placement. Chart: two small multiples.
4. **Damage without kills is real.** Chart: damage versus kills scatter, highlighting high-damage, low-kill players who set up teammates.
5. **Contribution styles exist.** Clustering gives archetypes (fragger, support, navigator, survivor). Chart: radar or parallel coordinates per archetype, plus how often each archetype sits on winning teams.
6. **Data caveats.** Missing teammates (observed squad size against mode capacity), non-standard modes, and cheater outliers (72 kills, 97K zero-movement players). Chart: observed-size histogram by mode.

Note: the figures above come from an earlier player-level scan. Confirm them on the team-level table before presenting.

## Plan

1. **Clean:** filter to standard modes, drop the bad row (missing `winPlacePerc`) and flagged outliers, de-duplicate the two copies of the file.
2. **EDA:** the six charts above, saved as reusable data for the dashboard.
3. **Feature engineering:**
   - Match-relative percentiles per player.
   - Per-capita team aggregates (mean, max, std, team size, mode).
   - Derived features: unconverted damage, revive rate against downs, kill share, engagement rate.
4. **Modelling** (grouped K-fold by `matchId`, scored on MAE and R²):
   - Ridge or GAM as the transparent anchor.
   - LightGBM as the strong model.
   - Compare with and without the leaky features.
5. **Attribution:** leave-one-out per member for the ML model, plus coefficients and SHAP for pillar weights. Distil these into the final linear rule and report how closely they agree.
6. **Allocation and evaluation:**
   - Build the rule.
   - Compare against kill-share, damage-share and equal split.
   - Test the axioms.
   - Run the style-parity and sensitivity analyses.
7. **Dashboard** (below).
8. **Slides:** 12 slides max (excluding title and appendix), with the dashboard as the live demo. Submitted as a PDF.

## Dashboard

Written as an editorial data piece, not a SaaS template.

- **Look:** warm paper-toned background, one ink color and one accent, serif headlines with a clean sans for labels, and real numbers set as typography. No gradient cards, no emoji icons, no purple.
- **Sections:**
  - Overview and the argument.
  - The data and its caveats.
  - Insights.
  - Match explorer: pick any real team and see its receipt, with kills-only shown alongside.
  - Sandbox: pillar-weight sliders and a τ dial, updating the split live.
  - Fairness scorecard.
  - Sensitivity heatmap.
  - Archetype view.
- **Stack:** Python precomputes everything to JSON, and a static page renders it with D3 or Plotly. It's fast, needs no server, and can be shared as a link.

## Open decision

Modelling is at **team level** (one row per team), not player level. The label is shared by teammates, so a player-level model would double-count.

## Data

PUBG Finish Placement Prediction (Kaggle), `train_V2.csv`: about 4.45M players, 2.0M teams, 29 columns. The copy in `data/` and `Copy of train_V2.csv` are byte-identical.
