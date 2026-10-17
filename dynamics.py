"""Phase 4: team dynamics + behavioural patterns (split B for anything model-fitted)."""
import pandas as pd, numpy as np, joblib, json, itertools, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
sns.set_theme(style="whitegrid")
mod = joblib.load("roles/gmm.joblib"); NAMES = mod["names"]; K = len(NAMES); ORDER = [NAMES[i] for i in range(K)]; M = [f"m{i}" for i in range(K)]
PAL = sns.color_palette("tab10", K); CP = dict(zip(ORDER, PAL))
FEATS = ["kills","damageDealt","DBNOs","assists","revives","heals","boosts","walkDistance","rideDistance"]
df = pd.read_parquet("data/scored.parquet")
Bd = df[df.split == "B"].copy(); J = {}
def save(p): plt.tight_layout(); plt.savefig(p, dpi=130); plt.close()

# ============ TEAM DYNAMICS ============
g = Bd.groupby("team")
T = g.agg(place=("winPlacePerc", "first"), size=("Id", "size"), ng=("numGroups", "first"), mt=("matchType", "first"), top10=("top10", "first"),
          topshare=("pts", "max"), walk_mean=("walkDistance", "mean"), walk_std=("walkDistance", "std"), walk_min=("walkDistance", "min"), walk_max=("walkDistance", "max"),
          kills=("kills", "sum"), revives=("revives", "sum"), **{f"c{i}": (f"m{i}", "sum") for i in range(K)})
T["cohesion"] = 1 - (T.walk_std / T.walk_mean.replace(0, np.nan)).clip(0, 2) / 2          # 1 = moved together
T["straggler"] = T.walk_min / T.walk_max.replace(0, np.nan)
T["gini_pts"] = g.pts.apply(lambda s: (np.abs(s.values[:, None] - s.values[None]).sum() / (2 * len(s) ** 2 * s.mean())))
T["kill_topshare"] = g.kills.apply(lambda s: s.max() / s.sum() if s.sum() > 0 else np.nan)
T4 = T[T["size"] == 4].copy()

# 1 team archetypes from composition
comp = T4[[f"c{i}" for i in range(K)]].div(4); comp.columns = ORDER
km = KMeans(6, n_init=10, random_state=0).fit(comp); T4["arch"] = km.labels_
cen = pd.DataFrame(km.cluster_centers_, columns=ORDER)
stat = T4.groupby("arch").agg(n=("place", "size"), place=("place", "mean"), top10=("top10", "mean"), kills=("kills", "mean"), revives=("revives", "mean"), topshare=("topshare", "mean"))
def nm(r):
    hi = r.sort_values(ascending=False); return " + ".join(hi.index[:2])
labels = {i: nm(cen.loc[i] - cen.mean()) for i in range(6)}   # two most over-represented roles vs the average squad
used = {}
for i in range(6):                                                        # disambiguate duplicates
    used[labels[i]] = used.get(labels[i], 0) + 1
    if used[labels[i]] > 1: labels[i] += f" #{used[labels[i]]}"
stat.index = [labels[i] for i in stat.index]; cen.index = [labels[i] for i in cen.index]; stat.to_csv("dynamics/team_archetypes.csv"); J["team_archetypes"] = stat.round(3).to_dict("index")
fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
sns.heatmap(cen, cmap="Blues", annot=True, fmt=".2f", ax=ax[0], annot_kws={"size": 7}); ax[0].set_title("Team archetype = average role mix")
stat.place.sort_values().plot.barh(ax=ax[1], color="#2f855a"); ax[1].set_title("Mean placement by team archetype")
stat.n.plot.barh(ax=ax[2], color="#2b6cb0"); ax[2].set_title("Squads per archetype"); save("dynamics/01_team_archetypes.png")

# 2 pairwise synergy (soft), controlling single role counts
Mm = Bd[Bd.team_size == 4].sort_values("team"); mm = Mm[M].values.reshape(-1, 4, K); tid = Mm.team.values[::4]
single = mm.sum(1); pair = np.zeros((len(mm), K, K))
for a, b in itertools.combinations(range(4), 2): pair += np.einsum("ni,nj->nij", mm[:, a], mm[:, b]) + np.einsum("ni,nj->nij", mm[:, b], mm[:, a])
iu = np.triu_indices(K); Xp = pair[:, iu[0], iu[1]]
y = T4.loc[tid, "place"].values; ng = T4.loc[tid, "ng"].values[:, None]
X = np.hstack([single, Xp, ng]); rd = Ridge(alpha=200).fit(X, y); cf = rd.coef_[K:K + len(iu[0])]
S = np.zeros((K, K)); S[iu] = cf; S = S + S.T - np.diag(np.diag(S))
plt.figure(figsize=(8, 6.5)); sns.heatmap(pd.DataFrame(S, index=ORDER, columns=ORDER), cmap="vlag", center=0, annot=True, fmt=".3f", annot_kws={"size": 7}); plt.title("Role-pair synergy: placement effect beyond individual roles (ridge)")
save("dynamics/02_pair_synergy.png"); J["pair_synergy"] = pd.DataFrame(S, index=ORDER, columns=ORDER).round(4).to_dict()

# 3 coverage: placement when squad has ≥1 of each role (hard) vs none
hard = Bd[Bd.team_size == 4].groupby("team").role_name.apply(set)
rows = []
for r in ORDER:
    has = hard.map(lambda s: r in s); p1, p0 = T4.loc[has.index[has], "place"], T4.loc[has.index[~has], "place"]
    rows.append(dict(role=r, with_role=p1.mean(), without=p0.mean(), lift=p1.mean() - p0.mean(), share=has.mean()))
cov = pd.DataFrame(rows).set_index("role"); cov.to_csv("dynamics/role_coverage.csv")
plt.figure(figsize=(9, 4.5)); cov.lift.plot.bar(color=[CP[n] for n in ORDER]); plt.axhline(0, color="k"); plt.title("Placement lift when squad contains ≥1 of the role (raw, confounded by survival)"); save("dynamics/03_role_coverage.png")

# 4 carry vs balance
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
for c, a, t_ in (("topshare", ax[0], "top player's share of points"), ("gini_pts", ax[1], "inequality of points (Gini)"), ("kill_topshare", ax[2], "top killer's share of kills")):
    b = pd.qcut(T4[c].rank(method="first"), 20, labels=False); gg = T4.groupby(b).agg(x=(c, "mean"), y=("place", "mean")); a.plot(gg.x, gg.y, marker="o", color="#b83280"); a.set_xlabel(t_); a.set_ylabel("mean placement")
ax[0].set_title("Do carry-heavy squads win?"); save("dynamics/04_carry_vs_balance.png")

# 5 cohesion
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
b = pd.qcut(T4.cohesion.rank(method="first"), 20, labels=False); gg = T4.groupby(b).agg(x=("cohesion", "mean"), y=("place", "mean")); ax[0].plot(gg.x, gg.y, marker="o", color="#2b6cb0"); ax[0].set_xlabel("cohesion (1 = identical walk distance)"); ax[0].set_title("Moving together vs placement")
b = pd.qcut(T4.straggler.rank(method="first"), 20, labels=False); gg = T4.groupby(b).agg(x=("straggler", "mean"), y=("place", "mean")); ax[1].plot(gg.x, gg.y, marker="o", color="#dd6b20"); ax[1].set_xlabel("min/max walk distance in squad"); ax[1].set_title("Straggler effect")
b = pd.cut(T4.cohesion, [0, .6, .75, .85, .92, 1.0]); T4.groupby(b, observed=True).revives.mean().plot.bar(ax=ax[2], color="#2f855a"); ax[2].set_title("Revives per squad by cohesion band"); save("dynamics/05_cohesion.png")

# 6 who really leads: rank-1 earner under ours vs kills-only
B4 = Bd[Bd.team_size == 4]; top_ours = B4.loc[B4.groupby("team").pts.idxmax()].set_index("team"); top_k = B4.loc[B4.groupby("team").kills.idxmax()].set_index("team")
diff = (top_ours.Id != top_k.Id.reindex(top_ours.index)).mean(); J["rank1_differs_from_top_killer"] = float(diff)
mx = B4[B4.groupby("team").kills.transform("sum") > 0]; share_top_k = (top_ours.role_name.value_counts(normalize=True)).reindex(ORDER)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5)); share_top_k.plot.bar(ax=ax[0], color=[CP[n] for n in ORDER]); ax[0].set_title(f"Role of the squad's #1 earner (differs from top killer in {diff:.0%} of squads)")
rk = B4.assign(rk=B4.groupby("team").pts.rank(ascending=False, method="first")).groupby("rk")[["kills", "revives", "assists", "heals", "damageDealt"]].mean(); (rk / rk.max()).plot.bar(ax=ax[1]); ax[1].set_title("Stats of 1st..4th ranked player (scaled)"); save("dynamics/06_who_leads.png")

# 7 team-role-mix triangle-ish: fragger/support/mobility composition vs placement (hexbin)
fr = (comp.Slayer + comp.Apex).values; sp = (comp.Medic + comp.Wingman).values
plt.figure(figsize=(7.5, 5.5)); hb = plt.hexbin(fr, sp, C=T4.place.values, gridsize=14, reduce_C_function=np.mean, cmap="viridis"); plt.colorbar(hb, label="mean placement"); plt.xlabel("fragger share (Slayer+Apex)"); plt.ylabel("support share (Medic+Wingman)"); plt.title("Squad composition landscape")
save("dynamics/07_composition_landscape.png")

# ============ BEHAVIOURAL PATTERNS ============
Bd["aggr"] = Bd[["q_kills", "q_damageDealt", "q_DBNOs"]].mean(1); Bd["sustain"] = Bd[["q_heals", "q_boosts"]].mean(1); Bd["mobil"] = Bd[["q_walkDistance", "q_rideDistance"]].mean(1); Bd["suppt"] = Bd[["q_assists", "q_revives"]].mean(1)
Bd["dmg_per_kill"] = np.where(Bd.kills > 0, Bd.damageDealt / Bd.kills.replace(0, np.nan), np.nan); Bd["hs_rate"] = np.where(Bd.kills > 0, Bd.headshotKills / Bd.kills.replace(0, np.nan), np.nan)
# 1 external validation: behaviours NOT used in clustering, by role
ext = Bd.groupby("role_name")[["hs_rate", "longestKill", "weaponsAcquired", "dmg_per_kill", "killStreaks", "swimDistance", "vehicleDestroys", "roadKills", "teamKills"]].mean().reindex(ORDER)
plt.figure(figsize=(11, 5)); sns.heatmap((ext - ext.mean()) / ext.std(), cmap="vlag", center=0, annot=ext.round(2), fmt="", annot_kws={"size": 7}); plt.title("External check: behaviours NOT used to build roles (colour = z-score across roles)")
save("behavior/01_external_validation.png"); ext.to_csv("behavior/external_means.csv")
# 2 quadrant map
Bd["kq"] = Bd.q_kills; qd = np.select([(Bd.kq > .6) & (Bd.winPlacePerc < .5), (Bd.kq > .6), (Bd.winPlacePerc >= .5)], ["Glass cannon", "Winning fragger", "Quiet survivor"], "Early ghost")
Bd["quad"] = qd; qs = pd.crosstab(Bd.role_name, Bd.quad, normalize="index").reindex(ORDER); J["quadrants"] = Bd.quad.value_counts(normalize=True).round(3).to_dict()
fig, ax = plt.subplots(1, 2, figsize=(15, 5)); s = Bd.sample(80000, random_state=0); hb = ax[0].hexbin(s.kq, s.winPlacePerc, gridsize=35, cmap="magma", bins="log"); ax[0].axvline(.6, color="w", ls="--"); ax[0].axhline(.5, color="w", ls="--"); ax[0].set_xlabel("kill percentile in lobby"); ax[0].set_ylabel("winPlacePerc"); ax[0].set_title("Aggression vs outcome")
qs.plot.bar(stacked=True, ax=ax[1]); ax[1].set_title("Outcome-style quadrant by role"); save("behavior/02_quadrants.png")
# 3 aggression/sustain tradeoff curves by mode family
Bd["fpp"] = np.where(Bd.matchType.str.contains("fpp"), "FPP", "TPP")
fig, ax = plt.subplots(1, 4, figsize=(20, 4.3))
for a, c, t_ in zip(ax, ["aggr", "sustain", "mobil", "suppt"], ["aggression", "sustain (heals/boosts)", "mobility", "support"]):
    for k_, d in Bd.groupby("fpp"):
        b = pd.cut(d[c], np.linspace(0, 1, 11), include_lowest=True); gg = d.groupby(b, observed=True).winPlacePerc.mean(); a.plot(np.linspace(.05, .95, len(gg)), gg.values, marker="o", label=k_)
    a.set_title(f"Placement vs {t_} percentile"); a.set_xlabel("percentile"); a.legend()
save("behavior/03_style_vs_placement.png")
# 4 teammate trade-offs: correlation of centred features
D = [f"d_{f}" for f in FEATS]; plt.figure(figsize=(8, 6.5)); sns.heatmap(Bd[D].corr().set_axis(FEATS).set_axis(FEATS, axis=1), cmap="vlag", center=0, annot=True, fmt=".2f", annot_kws={"size": 7}); plt.title("Within-squad trade-offs (correlation of 'more than teammates' scores)")
save("behavior/04_tradeoffs.png")
# 5 hybrids: top-2 role pairs
top2 = np.argsort(-Bd[M].values, 1)[:, :2]; hy = Bd.role_conf < .6; pc = np.zeros((K, K)); np.add.at(pc, (top2[hy.values, 0], top2[hy.values, 1]), 1); pc = pc + pc.T
plt.figure(figsize=(8, 6.5)); sns.heatmap(pd.DataFrame(pc / pc.sum(), index=ORDER, columns=ORDER), cmap="Blues", annot=True, fmt=".3f", annot_kws={"size": 7}); plt.title("Hybrid players: which two roles do they blend?"); save("behavior/05_hybrids.png")
J["hybrid_share"] = float(hy.mean())
# 6 efficiency map: damage/kill & hs rate by role violin
fig, ax = plt.subplots(1, 2, figsize=(15, 4.5)); sns.boxplot(data=Bd[Bd.kills >= 2], x="role_name", y="dmg_per_kill", order=ORDER, palette=CP, ax=ax[0], showfliers=False); ax[0].set_ylim(0, 300); ax[0].set_title("Damage per kill (kills≥2)")
sns.barplot(data=Bd[Bd.kills >= 2], x="role_name", y="hs_rate", order=ORDER, palette=CP, ax=ax[1], errorbar=None); ax[1].set_title("Headshot share of kills"); save("behavior/06_efficiency.png")
# 7 style profile spread: aggr vs sustain scatter coloured by role
s = Bd.sample(30000, random_state=1); plt.figure(figsize=(8, 6.5))
for n in ORDER: d = s[s.role_name == n]; plt.scatter(d.aggr, d.sustain, s=3, alpha=.35, color=CP[n], label=n)
plt.legend(markerscale=5, fontsize=8); plt.xlabel("aggression percentile"); plt.ylabel("sustain percentile"); plt.title("Behaviour map: aggression vs sustain"); save("behavior/07_behaviour_map.png")
# 8 role x mode behaviour: kills & damage by FPP/TPP
bm = Bd.groupby(["role_name", "fpp"])[["kills", "damageDealt", "heals", "walkDistance"]].mean().unstack().reindex(ORDER); bm.to_csv("behavior/role_by_perspective.csv")
fig, axs = plt.subplots(1, 4, figsize=(18, 4)); 
for a, c in zip(axs, ["kills", "damageDealt", "heals", "walkDistance"]): bm[c].plot.bar(ax=a); a.set_title(f"{c}: FPP vs TPP")
save("behavior/08_role_by_perspective.png")
# headline team facts
J["cohesion_corr_place"] = float(T4.cohesion.corr(T4.place)); J["topshare_corr_place"] = float(T4.topshare.corr(T4.place)); J["gini_corr_place"] = float(T4.gini_pts.corr(T4.place))
J["coverage_lift"] = cov.lift.round(3).to_dict(); J["ext"] = ext.round(3).to_dict("index")
json.dump(J, open("dynamics/summary.json", "w"), indent=1, default=float); print(json.dumps({k: J[k] for k in ["team_archetypes","rank1_differs_from_top_killer","quadrants","cohesion_corr_place","topshare_corr_place","gini_corr_place","coverage_lift","hybrid_share"]}, indent=1, default=float))
