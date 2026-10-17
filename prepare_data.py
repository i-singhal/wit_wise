"""Phase 1a: load raw PUBG data, filter to squad modes, drop suspect rows, derive team-level fields."""
import pandas as pd, numpy as np, json

RAW, OUT = "data/train_V2.csv", "data/clean.parquet"
SQUAD_MODES = ["squad", "squad-fpp", "normal-squad", "normal-squad-fpp"]

df = pd.read_csv(RAW)
log = {"raw_rows": len(df)}

df = df.dropna(subset=["winPlacePerc"]);                       log["after_dropna"] = len(df)
df = df[df.matchType.isin(SQUAD_MODES)];                       log["after_squad_modes"] = len(df)

df["totalDistance"] = df.walkDistance + df.rideDistance + df.swimDistance
flags = {
    "kills_no_movement":  (df.kills > 0) & (df.totalDistance == 0),
    "longestKill_gt_1000": df.longestKill > 1000,
    "roadKills_gt_10":     df.roadKills > 10,
    "weapons_gt_80":       df.weaponsAcquired > 80,
    "heals_gt_40":         df.heals > 40,
    "kills_gt_25":         df.kills > 25,
    "teamKills_gt_3":      df.teamKills > 3,
}
for k, m in flags.items(): log[f"flag_{k}"] = int(m.sum())
bad = np.logical_or.reduce(list(flags.values()))
log["flagged_union"] = int(bad.sum())
df = df[~bad]

# team (squad) key + size; drop whole teams that contain a removed player? -> no, keep data, but require size 2..4 teams
df["team"] = df.matchId + "_" + df.groupId
df["team_size"] = df.groupby("team").Id.transform("size")
log["teams_size_gt4_rows"] = int((df.team_size > 4).sum())
df = df[(df.team_size >= 2) & (df.team_size <= 4)]
log["after_team_size_2_4"] = len(df)
df["match_players"] = df.groupby("matchId").Id.transform("size")
df["match_teams"] = df.groupby("matchId").team.transform("nunique")
log["players"], log["teams"], log["matches"] = len(df), int(df.team.nunique()), int(df.matchId.nunique())

df.to_parquet(OUT, index=False)
json.dump(log, open("data/clean_log.json", "w"), indent=1)
print(json.dumps(log, indent=1))
