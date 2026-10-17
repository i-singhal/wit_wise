"""Phase 2: team-relative percentile features -> GMM (k=8) fit on split A -> soft role membership for every player."""
import pandas as pd, numpy as np, joblib, json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
sns.set_theme(style="whitegrid")
FEATS = ["kills","damageDealt","DBNOs","assists","revives","heals","boosts","walkDistance","rideDistance"]
D = [f"d_{f}" for f in FEATS]
K, REG = 8, 0.002
NAMES = {0:"Passenger", 1:"Survivor", 2:"Medic", 3:"Rotator", 4:"Flex", 5:"Wingman", 6:"Slayer", 7:"Apex"}  # set after reading centroids (k=8, seed 1)

df = pd.read_parquet("data/clean.parquet")
for f in FEATS:
    p = df.groupby("matchId")[f].rank(pct=True)                 # within-match percentile
    df[f"d_{f}"] = p - p.groupby(df.team).transform("mean")      # minus teammates' mean: role relative to own squad
m = df.matchId.drop_duplicates().sample(frac=1, random_state=42).values
df["split"] = np.where(df.matchId.isin(set(m[: len(m)//2])), "A", "B")
A, B = df[df.split=="A"], df[df.split=="B"]
XA = A[D].sample(250_000, random_state=0).values; TB = B[D].sample(60_000, random_state=5).values

# ---- k selection: held-out LL and seed stability
rows = []
for k in range(3, 11):
    gs = [GaussianMixture(k, covariance_type="full", n_init=2, random_state=s, reg_covar=REG, max_iter=300).fit(XA) for s in (1, 2)]
    rows.append(dict(k=k, ari=adjusted_rand_score(gs[0].predict(TB), gs[1].predict(TB)), ll_holdout=np.mean([g.score(TB) for g in gs]), bic=gs[0].bic(XA)))
R = pd.DataFrame(rows); R.to_csv("roles/k_selection.csv", index=False); print(R.round(3))
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(R.k, R.ll_holdout, marker="o"); ax[0].set_title("Held-out log-likelihood"); ax[0].axvline(K, color="r", ls="--")
ax[1].plot(R.k, R.ari, marker="o", color="#2f855a"); ax[1].set_title("Seed stability (ARI, 1 = identical)"); ax[1].axvline(K, color="r", ls="--")
ax[2].plot(R.k, R.bic, marker="o", color="#b83280"); ax[2].set_title("BIC (fit half)"); ax[2].axvline(K, color="r", ls="--")
for a in ax: a.set_xlabel("k (number of roles)")
plt.tight_layout(); plt.savefig("roles/01_k_selection.png", dpi=130); plt.close()

# ---- final model
g = GaussianMixture(K, covariance_type="full", n_init=5, random_state=1, reg_covar=REG, max_iter=300).fit(XA)
joblib.dump({"gmm": g, "feats": FEATS, "names": NAMES}, "roles/gmm.joblib")
cen = pd.DataFrame(g.means_, columns=FEATS); cen["w"] = g.weights_; print(cen.round(2))
pr = g.predict_proba(df[D].values)
for i in range(K): df[f"m{i}"] = pr[:, i]
df["role"] = pr.argmax(1); df["role_name"] = df.role.map(NAMES); df["role_conf"] = pr.max(1)
df.to_parquet("data/roles.parquet", index=False)

order = [NAMES[i] for i in range(K)]
cen.index = order
PAL = sns.color_palette("tab10", K); CP = {n: PAL[i] for i, n in enumerate(order)}

# 02 centroid heatmap
plt.figure(figsize=(10, 5)); sns.heatmap(cen[FEATS], cmap="vlag", center=0, annot=True, fmt=".2f")
plt.title("Role centroids: percentile advantage over own teammates"); plt.tight_layout(); plt.savefig("roles/02_centroids.png", dpi=130); plt.close()

# 03 radars
ang = np.linspace(0, 2*np.pi, len(FEATS), endpoint=False).tolist(); ang += ang[:1]
fig, axs = plt.subplots(2, 4, figsize=(18, 9), subplot_kw=dict(polar=True)); axs = axs.ravel()
for i, n in enumerate(order):
    v = cen.loc[n, FEATS].tolist(); v += v[:1]
    axs[i].plot(ang, v, color=CP[n]); axs[i].fill(ang, v, color=CP[n], alpha=.3); axs[i].set_ylim(-.3, .35)
    axs[i].set_xticks(ang[:-1]); axs[i].set_xticklabels(FEATS, fontsize=7); axs[i].set_title(f"{n}  ({g.weights_[i]:.0%})", pad=14)
plt.tight_layout(); plt.savefig("roles/03_radars.png", dpi=130); plt.close()

# 04 sizes + 05 hybrid-ness
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
df.role_name.value_counts().reindex(order).plot.bar(ax=ax[0], color=[CP[n] for n in order]); ax[0].set_title("Players by hard role")
df.role_conf.plot.hist(bins=40, ax=ax[1], color="#2b6cb0"); ax[1].set_title(f"Max membership (mean {df.role_conf.mean():.2f}; <0.6: {(df.role_conf<.6).mean():.1%})")
mem = df[[f"m{i}" for i in range(K)]].values; ent = -(mem*np.log(mem+1e-12)).sum(1)
pd.Series(ent).plot.hist(bins=40, ax=ax[2], color="#dd6b20"); ax[2].set_title("Membership entropy (higher = more hybrid)")
plt.tight_layout(); plt.savefig("roles/04_sizes_hybrid.png", dpi=130); plt.close()

# 06 role mix by mode and team size
fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
pd.crosstab(df.matchType, df.role_name, normalize="index")[order].plot.bar(stacked=True, ax=ax[0], color=[CP[n] for n in order]); ax[0].set_title("Role mix by match type"); ax[0].legend(fontsize=7, ncol=2)
pd.crosstab(df.team_size, df.role_name, normalize="index")[order].plot.bar(stacked=True, ax=ax[1], color=[CP[n] for n in order]); ax[1].set_title("Role mix by team size"); ax[1].legend().remove()
plt.tight_layout(); plt.savefig("roles/05_mix.png", dpi=130); plt.close()

# 07 PCA scatter
s = df.sample(40_000, random_state=0); pc = PCA(2).fit(df[D].sample(200_000, random_state=0).values); z = pc.transform(s[D].values)
plt.figure(figsize=(9, 7))
for i, n in enumerate(order): k_ = s.role.values == i; plt.scatter(z[k_, 0], z[k_, 1], s=3, alpha=.4, color=CP[n], label=n)
plt.legend(markerscale=5, fontsize=8); plt.title(f"Role space (PCA of team-relative features, {pc.explained_variance_ratio_.sum():.0%} variance)")
plt.tight_layout(); plt.savefig("roles/06_pca.png", dpi=130); plt.close()

# 08 raw stats per role (real units) + outcome
raw = df.groupby("role_name")[["kills","damageDealt","assists","revives","heals","boosts","walkDistance","rideDistance","winPlacePerc"]].mean().reindex(order)
raw.to_csv("roles/role_raw_means.csv")
fig, axs = plt.subplots(2, 5, figsize=(20, 7)); axs = axs.ravel()
for a, c in zip(axs, raw.columns): raw[c].plot.bar(ax=a, color=[CP[n] for n in order]); a.set_title(f"mean {c}"); a.tick_params(axis="x", labelsize=7)
axs[-1].axis("off"); plt.tight_layout(); plt.savefig("roles/07_raw_stats_by_role.png", dpi=130); plt.close()

# 09 team composition
t = df.groupby("team").agg(roles=("role_name", lambda s: tuple(sorted(s))), n_distinct=("role", "nunique"), size=("Id", "size"), place=("winPlacePerc", "first"))
top = t.roles.value_counts().head(12)
fig, ax = plt.subplots(1, 2, figsize=(17, 5))
top.sort_values().plot.barh(ax=ax[0], color="#2b6cb0"); ax[0].set_title("Most common squad compositions"); ax[0].tick_params(axis="y", labelsize=7)
x = t[t["size"] == 4].groupby("n_distinct").place.agg(["mean", "size"]); ax[1].bar(x.index, x["mean"], color="#2f855a"); ax[1].set_title("4-player squads: distinct roles vs mean placement")
for i, r in x.iterrows(): ax[1].text(i, r["mean"], f"n={int(r['size']):,}", ha="center", va="bottom", fontsize=8)
plt.tight_layout(); plt.savefig("roles/08_team_composition.png", dpi=130); plt.close()

# 10 role pair co-occurrence (lift)
pairs = np.zeros((K, K)); 
for r in t[t["size"] == 4].roles.sample(80_000, random_state=0):
    ix = [order.index(x) for x in r]
    for a in ix:
        for b in ix: pairs[a, b] += 1
marg = np.diag(pairs) / pairs.sum()
plt.figure(figsize=(8, 6.5)); sns.heatmap(pd.DataFrame(pairs / pairs.sum() / np.outer(marg, marg) , index=order, columns=order), cmap="vlag", center=1, annot=True, fmt=".2f", annot_kws={"size": 7})
plt.title("Role co-occurrence in a squad (lift vs chance)"); plt.tight_layout(); plt.savefig("roles/09_cooccurrence.png", dpi=130); plt.close()

# 11 split sanity: role shares A vs B
sh = pd.crosstab(df.split, df.role_name, normalize="index")[order]; print(sh.round(3))
print("hybrid share (conf<0.6):", round(float((df.role_conf < .6).mean()), 3))
json.dump({"k": K, "names": NAMES, "weights": dict(zip(order, g.weights_.round(3))), "mean_conf": float(df.role_conf.mean()), "hybrid_share": float((df.role_conf<.6).mean())}, open("roles/summary.json", "w"), indent=1, default=float)
