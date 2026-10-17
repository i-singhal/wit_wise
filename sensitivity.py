"""Sensitivity: re-run the whole allocation under alternative design choices and measure how much the 100-point splits move.
Roles are re-fit on split A only; benchmarks / importance on split B; everything is compared on the same B 4-player squads."""
import pandas as pd, numpy as np, joblib, json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import Ridge
sns.set_theme(style="whitegrid")
FEATS = ["kills","damageDealt","DBNOs","assists","revives","heals","boosts","walkDistance","rideDistance"]
Dc = [f"d_{f}" for f in FEATS]; Qc = [f"q_{f}" for f in FEATS]
mod = joblib.load("roles/gmm.joblib"); BASE_GMM = mod["gmm"]
df = pd.read_parquet("data/scored.parquet", columns=Dc + Qc + ["split", "team", "team_size", "winPlacePerc", "numGroups", "team_rank", "kills", "revives", "assists", "damageDealt", "pts_kills", "pts_fixed"])
A = df[df.split == "A"]; Ball = df[df.split == "B"]
B4 = Ball[Ball.team_size == 4].sort_values("team"); B4 = B4[B4.groupby("team").team.transform("size") == 4]
N = len(B4) // 4; print("B4 squads:", N, " B rows:", len(Ball))
XA = A[Dc].sample(150_000, random_state=0).values
gm_cache = {8: (BASE_GMM)}
def get_gmm(K, seed):
    if (K, seed) not in gm_cache and not (K == 8 and seed == 1):
        gm_cache[(K, seed)] = GaussianMixture(K, covariance_type="full", n_init=2, random_state=seed, reg_covar=0.002, max_iter=200).fit(XA)
    return BASE_GMM if (K == 8 and seed == 1) else gm_cache[(K, seed)]
def absl(d):
    q = lambda f: d[f"q_{f}"].values
    return .45 * (.5*q("damageDealt") + .3*q("DBNOs") + .2*q("kills")) + .35 * (.5*q("revives") + .35*q("assists") + .15*q("heals")) + .20 * (.6*q("walkDistance") + .4*q("rideDistance"))
ABS4 = absl(B4); Q4 = B4[Qc].values; QB = Ball[Qc].values; TOPrank = Ball.team_rank.values
cache_m = {}
def run(K=8, seed=1, floor=.03, topk=10, cap=.15, beta=.4, gamma=3, weights="centroid", transform="ecdf", jitter=None, jseed=0):
    g = get_gmm(K, seed); key = (K, seed)
    if key not in cache_m:
        cache_m[key] = (g.predict_proba(Ball[Dc].values), g.predict_proba(B4[Dc].values))
    mB, m4 = cache_m[key]; hard = mB.argmax(1)
    cen = pd.DataFrame(g.means_, columns=FEATS)
    W = (cen.clip(lower=0) + floor).values if weights == "centroid" else np.ones((K, len(FEATS)))
    W = W / W.sum(1, keepdims=True)
    if jitter: W = W * np.exp(np.random.default_rng(jseed).normal(0, jitter, W.shape)); W = W / W.sum(1, keepdims=True)
    S = np.zeros((len(B4), K))
    for r in range(K):
        sel = (hard == r) & (TOPrank <= topk)
        if sel.sum() < 50: sel = hard == r
        for j in range(len(FEATS)):
            b = np.sort(QB[sel, j]); v = Q4[:, j]
            if transform == "ecdf": sc = 0.5 * (np.searchsorted(b, v, "left") + np.searchsorted(b, v, "right")) / len(b)
            else: lo, hi = np.percentile(b, 10), np.percentile(b, 90); sc = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
            S[:, r] += W[r, j] * sc
    # importance from team model on B4 squads
    c = m4.reshape(N, 4, K).sum(1); y = B4.winPlacePerc.values.reshape(N, 4)[:, 0]; ng = B4.numGroups.values.reshape(N, 4)[:, 0]
    coef = Ridge(alpha=50).fit(np.column_stack([c, ng]), y).coef_[:K]; z = (coef - coef.mean()) / coef.std(); imp = 1 + cap * np.tanh(z)
    raw = ((1 - beta) * (m4 * S * imp).sum(1) + beta * ABS4) ** gamma
    raw = raw.reshape(N, 4); return 100 * raw / raw.sum(1, keepdims=True)
sq = lambda c: B4[c].values.reshape(N, 4)
KILLS = sq("kills"); ZK = KILLS == 0
def rk(a): return a.argsort(1).argsort(1)
def compare(P, base):
    r1, r2 = rk(P).astype(float), rk(base).astype(float); r1 -= r1.mean(1, keepdims=True); r2 -= r2.mean(1, keepdims=True)
    den = np.sqrt((r1**2).sum(1) * (r2**2).sum(1)); rho = np.where(den > 0, (r1 * r2).sum(1) / np.where(den > 0, den, 1), 1.0)
    return dict(mad=float(np.abs(P - base).mean()), rank_rho=float(rho.mean()), top_earner_changed=float((P.argmax(1) != base.argmax(1)).mean()),
                zero_kill_mean=float(P[ZK].mean()), top_share=float(P.max(1).mean()))
base = run(); rows = [dict(family="base", config="base (K=8, β=.4, γ=3, cap ±15%, top-10, ECDF)", **compare(base, base))]
def add(fam, name, **kw): print(fam, name, flush=True); rows.append(dict(family=fam, config=name, **compare(run(**kw), base)))
add("noise floor", "K=8 refit, different seed", K=8, seed=7)
for k in (5, 6, 7, 9, 10): add("number of roles K", f"K={k}", K=k, seed=1)
for b in (0, .2, .6, .8): add("absolute-lens weight β", f"β={b}", beta=b)
for g_ in (1, 2, 4, 5): add("contrast γ", f"γ={g_}", gamma=g_)
for c in (0, .05, .30): add("importance cap", f"cap ±{int(c*100)}%", cap=c)
for t in (5, 20): add("benchmark teams", f"top-{t} teams", topk=t)
add("rubric weights", "no floor (0)", floor=0); add("rubric weights", "floor 0.10", floor=.10); add("rubric weights", "uniform weights", weights="uniform")
add("transform", "linear p10–p90 instead of ECDF", transform="linear")
mc = [compare(run(jitter=.3, jseed=s), base) for s in range(8)]
rows.append(dict(family="rubric weights", config="random ±30% weight noise (mean of 8 draws)", **{k: float(np.mean([m[k] for m in mc])) for k in mc[0]}))
R = pd.DataFrame(rows); R.to_csv("validate/sensitivity.csv", index=False); print(R.round(3).to_string())
# chart
Rp = R[R.family != "base"].iloc[::-1]; pal = dict(zip(Rp.family.unique(), sns.color_palette("tab10", Rp.family.nunique())))
fig, ax = plt.subplots(1, 3, figsize=(20, 8), sharey=True)
for a, c, t in zip(ax, ["mad", "rank_rho", "top_earner_changed"], ["Mean |Δ points| per player (of 100)", "Within-squad rank agreement (Spearman, 1 = identical)", "Squads whose #1 earner changes"]):
    a.barh(Rp.config, Rp[c], color=[pal[f] for f in Rp.family]); a.set_title(t)
ax[1].set_xlim(.5, 1); plt.tight_layout(); plt.savefig("validate/01_sensitivity.png", dpi=130); plt.close()
json.dump(R.round(4).to_dict("records"), open("validate/sensitivity.json", "w"), indent=1)
