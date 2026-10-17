"""Phase 5: aggregate everything into one JSON, embed compressed charts + decision log, write dashboard.html (single file)."""
import pandas as pd, numpy as np, json, base64, io, glob, os, joblib, markdown
from PIL import Image
mod = joblib.load("roles/gmm.joblib"); NAMES = mod["names"]; K = len(NAMES); ORDER = [NAMES[i] for i in range(K)]
FEATS = ["kills","damageDealt","DBNOs","assists","revives","heals","boosts","walkDistance","rideDistance"]
M = [f"m{i}" for i in range(K)]; S = [f"s{i}" for i in range(K)]
df = pd.read_parquet("data/scored.parquet"); B = df[df.split == "B"].copy()
P = json.load(open("score/params.json")); H = json.load(open("score/headline.json")); DY = json.load(open("dynamics/summary.json"))
EDA = json.load(open("eda/findings.json")); RS = json.load(open("roles/summary.json")); CL = json.load(open("data/clean_log.json"))
D = {}
D["meta"] = dict(players=len(df), teams=int(df.team.nunique()), matches=int(df.matchId.nunique()), raw_rows=CL["raw_rows"], order=ORDER, feats=FEATS)
D["headline"] = H; D["params"] = P; D["eda"] = {k: EDA[k] for k in ["zero_share","placement_corr","top10_ratio","gini_mean","teams_with_zero_kill_player_share","players_zero_kills_share","zero_kill_players_with_support","zero_kill_players_with_damage_gt100"]}
D["funnel"] = {k: CL[k] for k in ["raw_rows","after_squad_modes","after_team_size_2_4"]}
D["ksel"] = pd.read_csv("roles/k_selection.csv").round(4).to_dict("list"); D["gamma"] = pd.read_csv("score/gamma_sensitivity.csv").round(3).to_dict("list")

# roles
cen = pd.DataFrame(mod["gmm"].means_, columns=FEATS); raw = pd.read_csv("roles/role_raw_means.csv", index_col=0); ext = pd.read_csv("behavior/external_means.csv", index_col=0)
sz = RS["weights"]
D["roles"] = [dict(name=n, share=sz[n], centroid=cen.iloc[i].round(3).tolist(), raw=raw.loc[n].round(3).to_dict(), ext=ext.loc[n].round(3).to_dict(), rubric=pd.Series(P["weights"][n]).round(3).to_dict() if n in P["weights"] else None) for i, n in enumerate(ORDER)]
D["importance"] = P["importance"]

# equal-share-relative points by role, three systems
B["eqs"] = 100 / B.team_size
rel = B.assign(ours=B.pts / B.eqs, kills=B.pts_kills / B.eqs, fixed=B.pts_fixed / B.eqs).groupby("role_name")[["kills", "fixed", "ours"]].mean().reindex(ORDER).round(3)
D["rel_points"] = rel.to_dict("list")

# team dynamics
D["dyn"] = {k: DY[k] for k in ["team_archetypes","pair_synergy","coverage_lift","rank1_differs_from_top_killer","quadrants","cohesion_corr_place","topshare_corr_place","gini_corr_place","hybrid_share"]}
g = B[B.team_size == 4].groupby("team")
T = g.agg(place=("winPlacePerc", "first"), topshare=("pts", "max"), wm=("walkDistance", "mean"), ws=("walkDistance", "std"), wmin=("walkDistance", "min"), wmax=("walkDistance", "max"))
T["cohesion"] = 1 - (T.ws / T.wm.replace(0, np.nan)).clip(0, 2) / 2; T["straggler"] = T.wmin / T.wmax.replace(0, np.nan)
def curve(x, y, n=15):
    d = pd.DataFrame({"x": x, "y": y}).dropna(); b = pd.qcut(d.x.rank(method="first"), n, labels=False); gg = d.groupby(b).mean(); return {"x": gg.x.round(3).tolist(), "y": gg.y.round(4).tolist()}
D["curves"] = dict(topshare=curve(T.topshare, T.place), cohesion=curve(T.cohesion, T.place), straggler=curve(T.straggler, T.place))
# behaviour
B["fpp"] = np.where(B.matchType.str.contains("fpp"), "FPP", "TPP")
B["aggr"] = B[["q_kills", "q_damageDealt", "q_DBNOs"]].mean(1); B["sustain"] = B[["q_heals", "q_boosts"]].mean(1); B["mobil"] = B[["q_walkDistance", "q_rideDistance"]].mean(1); B["suppt"] = B[["q_assists", "q_revives"]].mean(1)
bc = {}
for c in ["aggr", "sustain", "mobil", "suppt"]:
    bc[c] = {}
    for k, d in B.groupby("fpp"):
        b = pd.cut(d[c], np.linspace(0, 1, 11), include_lowest=True); gg = d.groupby(b, observed=True).winPlacePerc.mean(); bc[c][k] = gg.round(4).tolist()
D["style_curves"] = bc
B["quad"] = np.select([(B.q_kills > .6) & (B.winPlacePerc < .5), (B.q_kills > .6), (B.winPlacePerc >= .5)], ["Glass cannon", "Winning fragger", "Quiet survivor"], "Early ghost")
D["quad_by_role"] = pd.crosstab(B.role_name, B.quad, normalize="index").reindex(ORDER).round(3).to_dict("list")
top2 = np.argsort(-B[M].values, 1)[:, :2]; hy = (B.role_conf < .6).values
hp = pd.Series([" + ".join(sorted((NAMES[a], NAMES[b]))) for a, b in top2[hy]]).value_counts(normalize=True).head(10).round(3); D["hybrid_pairs"] = hp.to_dict()
D["team_corr"] = DY

# squads for explorer
B4 = B[B.team_size == 4].copy(); B4["delta"] = B4.pts - B4.pts_kills
tg = B4.groupby("team").agg(big=("delta", lambda s: s.abs().max()), k=("kills", "sum"), rank=("team_rank", "first"), nd=("role", "nunique"))
rng = np.random.default_rng(3)
pick = []
pick += [(t, "Biggest re-allocation") for t in tg[tg.k >= 5].sort_values("big", ascending=False).head(150).sample(35, random_state=1).index]
pick += [(t, "Top-3 finish") for t in tg[(tg["rank"] <= 3) & (tg.nd >= 3)].sample(30, random_state=2).index]
pick += [(t, "Typical squad") for t in tg[tg.nd >= 2].sample(30, random_state=4).index]
pick += [(t, "Early exit") for t in tg[(tg["rank"] >= 18)].sample(15, random_state=5).index]
sq = []
for t, tag in pick:
    d = B4[B4.team == t].sort_values("kills", ascending=False)
    sq.append(dict(id=t[:6], tag=tag, place=round(float(d.winPlacePerc.iloc[0]), 3), mode=d.matchType.iloc[0], players=[dict(
        role=r.role_name, a=round(float(r["abs"]), 3), m=[round(float(x), 3) for x in r[M]], s=[round(float(x), 3) for x in r[S]],
        stats={f: round(float(r[f]), 1) for f in FEATS}, pk=round(float(r.pts_kills), 1), pf=round(float(r.pts_fixed), 1)) for _, r in d.iterrows()]))
D["squads"] = sq

# images
def img(path, w=1100):
    im = Image.open(path).convert("RGB"); im.thumbnail((w, w)); b = io.BytesIO(); im.save(b, "JPEG", quality=82, optimize=True); return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
titles = {}
groups = [("EDA", "eda"), ("Roles", "roles"), ("Scoring", "score"), ("Team dynamics", "dynamics"), ("Behaviour", "behavior")]
D["gallery"] = [dict(group=gn, name=os.path.basename(p)[:-4], src=img(p)) for gn, d_ in groups for p in sorted(glob.glob(f"{d_}/*.png"))]
D["decisions_html"] = markdown.markdown(open("DECISIONS.md").read(), extensions=["tables"])

tpl = open("dashboard_template.html").read()
out = tpl.replace("/*__DATA__*/null", json.dumps(D, separators=(",", ":"), default=float))
open("dashboard.html", "w").write(out); print("dashboard.html", round(len(out) / 1e6, 2), "MB; squads:", len(sq), "images:", len(D["gallery"]))
