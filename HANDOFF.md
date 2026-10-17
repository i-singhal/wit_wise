# Handoff (state at push)

**Pipeline (run from repo root, in this order):** `prepare_data.py` → `eda.py` → `roles.py` → `score.py` → `dynamics.py` → `build_dashboard.py`.
Needs `data/train_V2.csv` (not in git). Python deps: pandas numpy scikit-learn matplotlib seaborn scipy pyarrow joblib markdown pillow.
All design decisions, evidence and caveats are in **DECISIONS.md** (read this first).

## Done
Data cleaning + 18 EDA charts · 8 roles via GMM (soft membership) · role benchmarks / importance / 100-point split · team dynamics + behaviour analysis (charts in `dynamics/`, `behavior/`) · decision log through Phase 4.

## NOT done / for the teammate
1. **Dashboard is unfinished and stale.** `dashboard.html` was built *before* the β absolute-lens change (D3.10) and with a broken Plotly CDN URL; `dashboard_template.html` already has the fix (Plotly 2.35.0) and the β slider, and `build_dashboard.py` already emits the `a` field. Just run `python build_dashboard.py` (needs `data/scored.parquet` from `score.py`) and eyeball every tab in a browser. The Behaviour tab and Squad explorer were only checked in screenshots before the CDN fix (charts rendered blank because of it).
2. **Sensitivity analysis written but NOT run:** `python sensitivity.py` (varies K, β, γ, importance cap, benchmark teams, rubric floor/uniform weights, transform; reports mean |Δpoints|, rank agreement, #1-earner changes). Takes a while (refits GMMs). Output goes to `validate/`.
3. **Not written yet:** explicit success/fairness definitions + metrics (budget=100, contributor coverage, role-parity ratio, dominance consistency), a held-out predictive-validity sanity check (RMSE/MAE/R² of team-level score vs placement; evaluate on split A, which has no placement leakage), final README, judges' talk track.
4. Math-branch files (`engine.py`, `app.py`, `test_run.py`) are untouched legacy (fixed 45/35/20 pillars, expects a `timeSurvived` column the data doesn't have).
5. Known limitations to state to judges: roles are per-match (no stable player IDs); learned role importance is confounded by survival (hence the ±15% cap); γ and β are policy knobs, not fitted.
