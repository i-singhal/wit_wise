"""Phase 2: within-match percentile features -> GMM on fit-half of matches -> soft role membership for everyone."""
import pandas as pd, numpy as np, json, joblib, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
sns.set_theme(style="whitegrid")

FEATS = ["kills","damageDealt","DBNOs","assists","revives","heals","boosts","walkDistance","rideDistance"]
df = pd.read_parquet("data/clean.parquet")

# within-match percentile rank (average ties) -> removes lobby/mode/duration scale effects, tames zero inflation
for f in FEATS:
    df[f"p_{f}"] = df.groupby("matchId")[f].rank(pct=True, method="average")
P = [f"p_{f}" for f in FEATS]

# split matches: A = fit roles, B = held out (benchmarks, importance, validation later)
m = df.matchId.drop_duplicates().sample(frac=1, random_state=42).values
A_ids = set(m[: len(m)//2]); df["split"] = np.where(df.matchId.isin(A_ids), "A", "B")
A, B = df[df.split == "A"], df[df.split == "B"]
XA = A[P].sample(250_000, random_state=0).values; XB = B[P].sample(100_000, random_state=0).values

res = []
for k in range(2, 10):
    g = GaussianMixture(k, covariance_type="full", n_init=2, random_state=0, max_iter=200).fit(XA)
    res.append(dict(k=k, bic=g.bic(XA), ll_fit=g.score(XA), ll_holdout=g.score(XB)))
    print(res[-1], flush=True)
R = pd.DataFrame(res); R.to_csv("roles/k_selection.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(R.k, R.bic, marker="o"); ax[0].set_title("BIC (lower = better)"); ax[0].set_xlabel("k")
ax[1].plot(R.k, R.ll_fit, marker="o", label="fit half"); ax[1].plot(R.k, R.ll_holdout, marker="s", label="held-out half"); ax[1].legend(); ax[1].set_title("Avg log-likelihood"); ax[1].set_xlabel("k")
plt.tight_layout(); plt.savefig("roles/01_k_selection.png", dpi=130); plt.close()
R.to_pickle("roles/k_selection.pkl")
