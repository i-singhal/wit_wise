"""Phase 3: role benchmarks (top-10 teams, split B) -> within-role percentile -> role importance -> 100-point split per team."""
import pandas as pd, numpy as np, joblib, json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.linear_model import Ridge
sns.set_theme(style="whitegrid")
FEATS = ["kills","damageDealt","DBNOs","assists","revives","heals","boosts","walkDistance","rideDistance"]
mod = joblib.load("roles/gmm.joblib"); NAMES = mod["names"]; K = len(NAMES); ORDER = [NAMES[i] for i in range(K)]
M = [f"m{i}" for i in range(K)]
df = pd.read_parquet("data/roles.parquet")
for f in FEATS: df[f"q_{f}"] = df.groupby("matchId")[f].rank(pct=True)           # ABSOLUTE within-match percentile (not teammate-centred)
Q = [f"q_{f}" for f in FEATS]
df["team_rank"] = np.round(1 + (1 - df.winPlacePerc) * (df.numGroups - 1)); df["top10"] = df.team_rank <= 10
B = df[df.split == "B"]

# ---- role rubric weights: defining (positive) centroid advantages count most, floor keeps every stat in play
cen = pd.DataFrame(mod["gmm"].means_, columns=FEATS)
W = (cen.clip(lower=0) + 0.03); W = W.div(W.sum(1), axis=0); W.index = ORDER
# ---- benchmark: ECDF of each stat among top-10-team players of that role (hard role), split B
bench = {}
for r in range(K):
    sub = B[(B.role == r) & B.top10]
    bench[r] = {f: np.sort(sub[f"q_{f}"].values) for f in FEATS}
    print(NAMES[r], "benchmark n =", len(sub))
def role_score(r, d):
    s = np.zeros(len(d))
    for j, f in enumerate(FEATS):
        v = d[f"q_{f}"].values; b = bench[r][f]; s += W.iloc[r, j] * (0.5 * (np.searchsorted(b, v, side="left") + np.searchsorted(b, v, side="right")) / len(b))   # mid-rank: ties no longer inflate
    return s
S = np.column_stack([role_score(r, df) for r in range(K)])            # player x role percentile score (0-1)
for i in range(K): df[f"s{i}"] = S[:, i]
mem = df[M].values
df["perf"] = (mem * S).sum(1)                                          # membership-weighted within-role performance

# ---- role importance: team model on split B, shrunk
t = B.groupby("team").agg(place=("winPlacePerc", "first"), size=("Id", "size"), ng=("numGroups", "first"), **{f"c{i}": (f"m{i}", "sum") for i in range(K)})
t = t[t["size"] == 4]
X = t[[f"c{i}" for i in range(K)]].sub(t["size"].values[:, None] / K * 0 , axis=0)  # role counts (soft)
rd = Ridge(alpha=50).fit(X.assign(ng=t.ng), t.place); coef = pd.Series(rd.coef_[:K], index=ORDER)
z = (coef - coef.mean()) / coef.std()
CAP = 0.15; imp = (1 + CAP * np.tanh(z)).round(3)                      # shrink: importance in [0.85, 1.15]
print(pd.DataFrame({"coef": coef.round(4), "z": z.round(2), "importance": imp}))
IMP = imp.values
GAMMA = 3.0   # contrast: >1 widens gaps between strong and weak performers; exposed as a slider in the dashboard
df["raw"] = (mem * S * IMP).sum(1) ** GAMMA
df["raw_noimp"] = df.perf ** GAMMA
for c, name in (("raw", "pts"), ("raw_noimp", "pts_noimp")):
    df[name] = 100 * df[c] / df.groupby("team")[c].transform("sum")

# ---- baselines
tk = df.groupby("team").kills.transform("sum"); df["pts_kills"] = np.where(tk > 0, 100 * df.kills / tk.replace(0, np.nan), 100 / df.team_size)
def sn(c): s = df.groupby("team")[c].transform("sum"); return np.where(s > 0, df[c] / s.replace(0, np.nan), 1 / df.team_size)
df["totalDistance"] = df.walkDistance + df.rideDistance
comb = .5*sn("damageDealt") + .3*sn("DBNOs") + .2*sn("kills"); sup = .5*sn("revives") + .35*sn("assists") + .15*sn("heals"); tac = sn("totalDistance")
fx = .45*comb + .35*sup + .20*tac; df["pts_fixed"] = 100 * fx / pd.Series(fx, index=df.index).groupby(df.team).transform("sum")   # Math-branch pillars, distance as survival proxy
df.to_parquet("data/scored.parquet", index=False)
json.dump({"importance": imp.to_dict(), "coef": coef.round(5).to_dict(), "weights": W.round(3).to_dict("index")}, open("score/params.json", "w"), indent=1)

# ================= validation charts (split B only) =================
Bd = df[df.split == "B"]; PAL = sns.color_palette("tab10", K); CP = dict(zip(ORDER, PAL))
# 01 rubric weights + importance
fig, ax = plt.subplots(1, 2, figsize=(16, 4.8))
sns.heatmap(W, cmap="Blues", annot=True, fmt=".2f", ax=ax[0], annot_kws={"size": 7}); ax[0].set_title("Role rubric: weight each stat carries")
imp.plot.bar(ax=ax[1], color=[CP[n] for n in ORDER]); ax[1].axhline(1, color="k"); ax[1].set_ylim(.8, 1.2); ax[1].set_title("Role importance multiplier (shrunk to ±15%)")
plt.tight_layout(); plt.savefig("score/01_rubric_importance.png", dpi=130); plt.close()
# 02 mean points by role, three systems
byr = Bd.groupby("role_name")[["pts", "pts_kills", "pts_fixed"]].mean().reindex(ORDER)
byr.plot.bar(figsize=(10, 4.5), color=["#2f855a", "#e53e3e", "#718096"]); plt.axhline(25, color="k", ls="--", lw=.8); plt.title("Mean points by role (4-player squads would average 25)"); plt.ylabel("points")
plt.tight_layout(); plt.savefig("score/02_points_by_role.png", dpi=130); plt.close()
# 03 zero-kill players
z0 = Bd[Bd.kills == 0]; v = pd.DataFrame({"kills-only": z0.pts_kills, "fixed pillars": z0.pts_fixed, "ours": z0.pts})
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5)); v.plot.hist(bins=40, alpha=.6, ax=ax[0], range=(0, 60)); ax[0].set_title("Points for zero-kill players")
v.mean().plot.bar(ax=ax[1], color=["#e53e3e", "#718096", "#2f855a"]); ax[1].set_title("Mean points, zero-kill players")
plt.tight_layout(); plt.savefig("score/03_zero_kill_players.png", dpi=130); plt.close()
# 04 delta distribution + concentration
Bd4 = Bd[Bd.team_size == 4]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
(Bd.pts - Bd.pts_kills).plot.hist(bins=60, ax=ax[0], color="#2b6cb0", range=(-60, 60)); ax[0].set_title("Ours − kills-only (pts)")
for c, l, col in (("pts", "ours", "#2f855a"), ("pts_kills", "kills-only", "#e53e3e"), ("pts_fixed", "fixed pillars", "#718096")):
    Bd4.groupby("team")[c].max().plot.hist(bins=40, ax=ax[1], alpha=.5, label=l, color=col)
ax[1].legend(); ax[1].set_title("Top player's share of 100 (4-player squads)")
sns.heatmap(Bd4.assign(rk=Bd4.groupby("team").pts.rank(ascending=False, method="first")).groupby(["rk", "role_name"]).size().unstack().reindex(columns=ORDER).pipe(lambda d: d.div(d.sum(1), axis=0)), cmap="Blues", ax=ax[2], annot=True, fmt=".2f", annot_kws={"size": 7}); ax[2].set_title("Who earns rank 1–4 in a squad (role share)")
plt.tight_layout(); plt.savefig("score/04_delta_concentration.png", dpi=130); plt.close()
# 05 within-role fairness: percentile score distribution per role
plt.figure(figsize=(11, 4.5)); sns.boxplot(data=Bd.assign(p=(Bd[M].values * S[Bd.index.map(lambda x: 0) if False else slice(None)][:0].sum(0) if False else 0)), x="role_name", y="perf", order=ORDER, palette=CP) if False else None
sns.boxplot(data=Bd, x="role_name", y="perf", order=ORDER, palette=CP); plt.title("Within-role performance score (0–1), by hard role: every role spans the range"); plt.tight_layout(); plt.savefig("score/05_perf_by_role.png", dpi=130); plt.close()
# 06 sensitivity: importance on/off, rank agreement
rho = Bd.groupby("team").apply(lambda g: g.pts.corr(g.pts_noimp, method="spearman") if len(g) > 2 else np.nan).dropna() if False else None
sh = (Bd.pts - Bd.pts_noimp).abs(); print("mean |pts - pts_noimp|:", round(float(sh.mean()), 3), "max:", round(float(sh.max()), 2))
# 07 case study: pick teams to show shift
Bd4 = Bd4.assign(delta=Bd4.pts - Bd4.pts_kills)
cs = Bd4.groupby("team").agg(big=("delta", lambda s: s.abs().max()), k=("kills", "sum")); case = cs[(cs.k >= 6)].sort_values("big", ascending=False).head(1).index[0]
g = Bd4[Bd4.team == case].sort_values("kills", ascending=False)
x = np.arange(len(g)); plt.figure(figsize=(8, 4.5)); plt.bar(x - .2, g.pts_kills, .4, label="kills-only", color="#e53e3e"); plt.bar(x + .2, g.pts, .4, label="ours", color="#2f855a")
plt.xticks(x, [f"{r}\nK{int(k)} R{int(rv)} A{int(a)}" for r, k, rv, a in zip(g.role_name, g.kills, g.revives, g.assists)]); plt.legend(); plt.title("Case study squad: biggest re-allocation"); plt.tight_layout(); plt.savefig("score/06_case_study.png", dpi=130); plt.close()

# 08 gamma sensitivity
gs = [1, 2, 3, 4, 5]; r_ = []
base = (Bd4[M].values * S[Bd4.index] * IMP).sum(1) if False else None
bs = Bd4.copy(); bs["lin"] = (bs[M].values * np.column_stack([bs[f"s{i}"] for i in range(K)]) * IMP).sum(1)
for gm in gs:
    pp = 100 * bs.lin ** gm / (bs.lin ** gm).groupby(bs.team).transform("sum")
    r_.append(dict(gamma=gm, top_share=float(pp.groupby(bs.team).max().mean()), zero_kill_mean=float(pp[bs.kills == 0].mean()), passenger_mean=float(pp[bs.role_name == "Passenger"].mean()), apex_mean=float(pp[bs.role_name == "Apex"].mean())))
G = pd.DataFrame(r_).set_index("gamma"); print(G.round(2)); G.to_csv("score/gamma_sensitivity.csv")
ax = G.plot(marker="o", figsize=(9, 4.5)); ax.axhline(bs.groupby("team").pts_kills.max().mean(), color="#e53e3e", ls="--", lw=.8); ax.set_title("Contrast exponent sensitivity (dashed: kills-only top share)"); plt.tight_layout(); plt.savefig("score/07_gamma_sensitivity.png", dpi=130); plt.close()

# headline numbers
H = {"zero_kill_mean_pts": {"kills_only": float(z0.pts_kills.mean()), "fixed": float(z0.pts_fixed.mean()), "ours": float(z0.pts.mean())},
     "zero_kill_pts_gt_10_share": {"kills_only": float((z0.pts_kills > 10).mean()), "ours": float((z0.pts > 10).mean())},
     "top_share_mean_4p": {"ours": float(Bd4.groupby("team").pts.max().mean()), "kills_only": float(Bd4.groupby("team").pts_kills.max().mean()), "fixed": float(Bd4.groupby("team").pts_fixed.max().mean())},
     "importance_effect_mean_abs": float(sh.mean()), "points_by_role": byr.round(2).to_dict("index")}
json.dump(H, open("score/headline.json", "w"), indent=1); print(json.dumps(H, indent=1))
