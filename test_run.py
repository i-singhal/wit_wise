import pandas as pd
from engine import calculate_fair_share

test_squad = pd.DataFrame([
    {"player": "P1_Fragger", "kills": 8, "damageDealt": 480, "DBNOs": 3, "assists": 0, "revives": 0, "heals": 1, "timeSurvived": 1350, "walkDistance": 1100, "rideDistance": 200},
    {"player": "P2_Support_Medic", "kills": 1, "damageDealt": 680, "DBNOs": 2, "assists": 5, "revives": 3, "heals": 7, "timeSurvived": 1500, "walkDistance": 1300, "rideDistance": 500},
    {"player": "P3_Scout_IGL", "kills": 2, "damageDealt": 310, "DBNOs": 1, "assists": 2, "revives": 1, "heals": 2, "timeSurvived": 1550, "walkDistance": 2800, "rideDistance": 1800},
    {"player": "P4_Anchor_Flex", "kills": 0, "damageDealt": 210, "DBNOs": 1, "assists": 3, "revives": 1, "heals": 2, "timeSurvived": 1280, "walkDistance": 950, "rideDistance": 300}
])

res = calculate_fair_share(test_squad)
print("=== TEAM WISE: FAIR SHARE REWARD ALLOCATION ===")
print(res[['player', 'kills', 'damageDealt', 'revives', 'baseline_pts', 'fair_share_pts', 'delta']].to_string(index=False))
print(f"Total Fair Share Points Sum: {res['fair_share_pts'].sum():.2f}")
