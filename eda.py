"""Phase 1b: exploratory analysis. Every chart saved to eda/ and summarized in eda/findings.json."""
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns, json

sns.set_theme(style="whitegrid", context="notebook")
PAL = ["#2b6cb0", "#dd6b20", "#2f855a", "#b83280", "#718096"]
df = pd.read_parquet("data/clean.parquet")
rng = np.random.default_rng(0)
samp = df.sample(200_000, random_state=0)
F = {}
def save(name):
    plt.tight_layout(); plt.savefig(f"eda/{name}.png", dpi=130); plt.close()

# 01 match type mix, team size mix
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
df.matchType.value_counts().plot.bar(ax=ax[0], color=PAL[0]); ax[0].set_title("Players by match type")
df.drop_duplicates("team").team_size.value_counts().sort_index().plot.bar(ax=ax[1], color=PAL[1]); ax[1].set_title("Teams by size")
df.drop_duplicates("matchId").match_teams.plot.hist(bins=30, ax=ax[2], color=PAL[2]); ax[2].set_title("Teams per match")
save("01_composition")

# 02 target distribution
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
df.winPlacePerc.plot.hist(bins=50, ax=ax[0], color=PAL[0]); ax[0].set_title("winPlacePerc (player rows)")
df.drop_duplicates("team").winPlacePerc.plot.hist(bins=50, ax=ax[1], color=PAL[1]); ax[1].set_title("winPlacePerc (team rows)")
save("02_target")

# 03 stat distributions (log1p), zero share annotated
stats = ["kills","damageDealt","DBNOs","assists","revives","heals","boosts","walkDistance","rideDistance","swimDistance","weaponsAcquired","headshotKills","longestKill","killStreaks"]
fig, axs = plt.subplots(3, 5, figsize=(20, 10)); axs = axs.ravel()
zero = {}
for a, s in zip(axs, stats):
    zero[s] = float((df[s] == 0).mean())
    np.log1p(samp[s]).plot.hist(bins=40, ax=a, color=PAL[0]); a.set_title(f"log1p({s})  zeros={zero[s]:.0%}", fontsize=10)
axs[-1].axis("off"); save("03_stat_distributions"); F["zero_share"] = zero

# 04 correlation heatmap
corr_cols = stats + ["killPlace","matchDuration","winPlacePerc"]
plt.figure(figsize=(12, 10)); sns.heatmap(samp[corr_cols].corr(), cmap="vlag", center=0, annot=True, fmt=".2f", annot_kws={"size": 7})
plt.title("Correlation matrix (200k sample)"); save("04_correlation")

# 05 stat vs placement (binned means)
fig, axs = plt.subplots(2, 4, figsize=(18, 8)); axs = axs.ravel()
for a, s in zip(axs, ["kills","damageDealt","walkDistance","rideDistance","heals","boosts","revives","assists"]):
    q = samp[s].clip(upper=samp[s].quantile(.99))
    b = pd.qcut(q.rank(method="first"), 20, labels=False)
    g = samp.groupby(b).agg(x=(s, "mean"), y=("winPlacePerc", "mean"))
    a.plot(g.x, g.y, marker="o", color=PAL[1]); a.set_title(f"{s} vs placement"); a.set_xlabel(s); a.set_ylabel("mean winPlacePerc")
save("05_stats_vs_placement")

# 06 placement correlation ranking
cor = samp[stats + ["killPlace"]].corrwith(samp.winPlacePerc).sort_values()
plt.figure(figsize=(7, 6)); cor.plot.barh(color=[PAL[1] if v < 0 else PAL[0] for v in cor]); plt.title("Correlation with winPlacePerc")
save("06_placement_corr"); F["placement_corr"] = cor.round(3).to_dict()

# 07 hexbins
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].hexbin(np.log1p(samp.walkDistance), np.log1p(samp.rideDistance), gridsize=40, cmap="Blues", bins="log"); ax[0].set_title("walk vs ride (log1p)"); ax[0].set_xlabel("walk"); ax[0].set_ylabel("ride")
ax[1].hexbin(samp.kills.clip(upper=15), samp.damageDealt.clip(upper=1500), gridsize=40, cmap="Oranges", bins="log"); ax[1].set_title("kills vs damage"); ax[1].set_xlabel("kills"); ax[1].set_ylabel("damage")
ax[2].hexbin(samp.heals.clip(upper=15), samp.boosts.clip(upper=15), gridsize=30, cmap="Greens", bins="log"); ax[2].set_title("heals vs boosts"); ax[2].set_xlabel("heals"); ax[2].set_ylabel("boosts")
save("07_hexbins")

# 08 match duration
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
df.drop_duplicates("matchId").matchDuration.plot.hist(bins=50, ax=ax[0], color=PAL[2]); ax[0].set_title("Match duration (s)")
sns.boxplot(data=samp, x="matchType", y="matchDuration", ax=ax[1], color=PAL[4]); ax[1].set_title("Duration by mode"); ax[1].tick_params(axis="x", rotation=20)
save("08_duration")

# ---- team-level motivation: how unfair is kills-only? ----
t = df.groupby("team").agg(size=("Id","size"), place=("winPlacePerc","first"), kills=("kills","sum"), kmax=("kills","max"),
                           dmg=("damageDealt","sum"), revives=("revives","sum"), assists=("assists","sum"), heals=("heals","sum"),
                           walk=("walkDistance","sum"), ride=("rideDistance","sum"), dbnos=("DBNOs","sum"))
t["top_kill_share"] = np.where(t.kills > 0, t.kmax / t.kills.replace(0, np.nan), np.nan)

# 09 kill concentration
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
t.top_kill_share.dropna().plot.hist(bins=30, ax=ax[0], color=PAL[3]); ax[0].set_title("Top killer's share of team kills")
sns.heatmap(pd.crosstab(t["size"], t.kills.clip(upper=12), normalize="index"), cmap="Blues", ax=ax[1]); ax[1].set_title("Team kills distribution by size")
t.kills.clip(upper=25).plot.hist(bins=26, ax=ax[2], color=PAL[0]); ax[2].set_title("Team kills")
save("09_kill_concentration")
F["teams_with_zero_kill_player_share"] = float(df.groupby("team").kills.apply(lambda s: (s == 0).any()).mean())
F["players_zero_kills_share"] = float((df.kills == 0).mean())

# 10 invisible contributors: zero-kill players
zk = df[df.kills == 0]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for a, s, c in zip(ax, ["revives","assists","heals"], PAL):
    zk[s].clip(upper=8).value_counts(normalize=True).sort_index().plot.bar(ax=a, color=c); a.set_title(f"Zero-kill players: {s}")
save("10_zero_kill_contributors")
F["zero_kill_players_with_support"] = float(((zk.revives > 0) | (zk.assists > 0)).mean())
F["zero_kill_players_with_damage_gt100"] = float((zk.damageDealt > 100).mean())

# 11 team stat vs team placement (the "what wins" view)
fig, axs = plt.subplots(2, 4, figsize=(18, 8)); axs = axs.ravel()
for a, s in zip(axs, ["kills","dmg","dbnos","revives","assists","heals","walk","ride"]):
    x = t[s].clip(upper=t[s].quantile(.99)); b = pd.qcut(x.rank(method="first"), 20, labels=False)
    g = t.groupby(b).agg(x=(s, "mean"), y=("place", "mean")); a.plot(g.x, g.y, marker="o", color=PAL[0]); a.set_title(f"team {s} vs placement"); a.set_xlabel(s)
save("11_team_stats_vs_placement")

# 12 within-team inequality: Gini of damage & kills
def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x); s = x.sum()
    return 0.0 if s == 0 else (2 * np.sum((np.arange(1, n + 1)) * x) / (n * s)) - (n + 1) / n
gt = df[df.team_size == 4].groupby("team").agg(gk=("kills", gini), gd=("damageDealt", gini), gh=("heals", gini), gw=("walkDistance", gini), gr=("revives", gini))
gt = gt.sample(min(len(gt), 60000), random_state=0)
plt.figure(figsize=(8, 4.5)); sns.violinplot(data=gt.rename(columns={"gk":"kills","gd":"damage","gh":"heals","gw":"walk","gr":"revives"}), palette=PAL, inner="quartile")
plt.title("Within-squad inequality (Gini), 4-player squads"); save("12_within_team_gini"); F["gini_mean"] = gt.mean().round(3).to_dict()

# 13 winners vs losers: stat means top-10 vs rest, per-stat ratio
df["team_rank"] = np.round(1 + (1 - df.winPlacePerc) * (df.numGroups - 1))
df["top"] = df.team_rank <= 10
ratio = (df[df.top][stats].mean() / df[~df.top][stats].mean()).sort_values()
plt.figure(figsize=(7, 6)); ratio.plot.barh(color=PAL[2]); plt.axvline(1, color="k"); plt.title("Mean stat: top-10 teams / others")
save("13_top10_vs_rest"); F["top10_ratio"] = ratio.round(2).to_dict(); F["top10_player_share"] = float(df.top.mean())

# 14 survival confound: stats by placement decile
df["dec"] = pd.cut(df.winPlacePerc, np.linspace(0, 1, 11), include_lowest=True, labels=False)
d = df.groupby("dec")[["kills","damageDealt","walkDistance","heals","boosts","revives"]].mean()
(d / d.max()).plot(figsize=(9, 5), marker="o"); plt.title("Stats (scaled) by placement decile — survivorship effect"); plt.xlabel("placement decile (9 = best)")
save("14_survivorship_confound")

# 15 crude archetype hint: scatter of damage share vs support/mobility share, coloured by placement
tt = df[df.team_size == 4].sample(60000, random_state=1)
tm = df[df.team.isin(tt.team)].groupby("team")
tt = df[df.team.isin(tt.team)].copy()
tt["dmg_share"] = tt.damageDealt / tt.groupby("team").damageDealt.transform("sum").replace(0, np.nan)
tt["move_share"] = tt.totalDistance / tt.groupby("team").totalDistance.transform("sum").replace(0, np.nan)
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
h = ax[0].hexbin(tt.dmg_share, tt.move_share, gridsize=40, cmap="viridis", bins="log"); ax[0].set_xlabel("share of team damage"); ax[0].set_ylabel("share of team distance"); ax[0].set_title("Role space hint: damage vs movement share")
sup = (tt.revives + tt.assists + tt.heals * 0.3)
ax[1].hexbin(tt.dmg_share, sup / tt.groupby("team").apply(lambda g: (g.revives + g.assists + g.heals * 0.3).sum()).reindex(tt.team).values, gridsize=40, cmap="magma", bins="log")
ax[1].set_xlabel("share of team damage"); ax[1].set_ylabel("share of team support actions"); ax[1].set_title("damage vs support share")
save("15_role_space_hint")

# 16 mode comparison
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, s in zip(ax, ["kills","damageDealt","walkDistance"]):
    sns.barplot(data=df.sample(300000, random_state=2), x="matchType", y=s, ax=a, color=PAL[0], errorbar=None); a.set_title(f"mean {s} by mode"); a.tick_params(axis="x", rotation=20)
save("16_mode_comparison")

# 17 team size effect on placement & stats
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, s in zip(ax, ["winPlacePerc","kills","revives"]):
    sns.barplot(data=df.sample(300000, random_state=3), x="team_size", y=s, ax=a, color=PAL[2], errorbar=None); a.set_title(f"mean {s} by team size")
save("17_team_size")

# 18 cleaning waterfall
lg = json.load(open("data/clean_log.json"))
steps = ["raw_rows","after_dropna","after_squad_modes","after_team_size_2_4"]
plt.figure(figsize=(8, 4)); plt.bar(range(len(steps)), [lg[s] for s in steps], color=PAL[0]); plt.xticks(range(len(steps)), steps, rotation=15); plt.title("Rows retained through cleaning")
save("18_cleaning_funnel")

F["rows"], F["teams"], F["matches"] = len(df), int(df.team.nunique()), int(df.matchId.nunique())
json.dump(F, open("eda/findings.json", "w"), indent=1, default=float)
print(json.dumps(F, indent=1, default=float))
