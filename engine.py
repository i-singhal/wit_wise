import pandas as pd
import numpy as np

def calculate_fair_share(df_squad, weights=None):
    """
    df_squad: DataFrame containing a single squad's match data with columns:
      ['player', 'kills', 'damageDealt', 'DBNOs', 'assists', 'revives', 'heals', 'timeSurvived', 'walkDistance', 'rideDistance']
    weights: dict of pillar and feature weights
    """
    if weights is None:
        weights = {
            'combat_w': 0.45,
            'support_w': 0.35,
            'tactical_w': 0.20,
            'combat_sub': {'damage': 0.50, 'dbno': 0.30, 'kills': 0.20},
            'support_sub': {'revives': 0.50, 'assists': 0.35, 'heals': 0.15},
            'tactical_sub': {'survival': 0.60, 'distance': 0.40}
        }
    
    df = df_squad.copy()
    
    # Total distance
    if 'totalDistance' not in df.columns:
        df['totalDistance'] = df.get('walkDistance', 0) + df.get('rideDistance', 0)
    if 'timeSurvived' not in df.columns and 'survival' in df.columns:
        df['timeSurvived'] = df['survival']
    if 'damageDealt' not in df.columns and 'damage' in df.columns:
        df['damageDealt'] = df['damage']

    # Anti-gaming caps (e.g. blue zone heal farming cap)
    df['heals_capped'] = np.minimum(df.get('heals', 0), 12)

    # Sum normalization within the squad
    def squad_norm(series):
        s = series.sum()
        return series / s if s > 0 else pd.Series([1.0 / len(series)] * len(series), index=series.index)

    # 1. Combat Impact
    c_sub = weights['combat_sub']
    combat = (c_sub['damage'] * squad_norm(df['damageDealt']) +
              c_sub['dbno'] * squad_norm(df.get('DBNOs', 0)) +
              c_sub['kills'] * squad_norm(df['kills']))

    # 2. Support & Teamwork
    s_sub = weights['support_sub']
    support = (s_sub['revives'] * squad_norm(df.get('revives', 0)) +
               s_sub['assists'] * squad_norm(df.get('assists', 0)) +
               s_sub['heals'] * squad_norm(df['heals_capped']))

    # 3. Tactical & Mobility
    t_sub = weights['tactical_sub']
    tactical = (t_sub['survival'] * squad_norm(df['timeSurvived']) +
                t_sub['distance'] * squad_norm(df['totalDistance']))

    # Composite Raw Score
    raw = (weights['combat_w'] * combat +
           weights['support_w'] * support +
           weights['tactical_w'] * tactical)

    # Fixed 100-Point Allocation
    df['fair_share_pts'] = (raw / raw.sum()) * 100

    # Baseline: Pure Kills
    total_kills = df['kills'].sum()
    if total_kills > 0:
        df['baseline_pts'] = (df['kills'] / total_kills) * 100
    else:
        df['baseline_pts'] = 100.0 / len(df)

    df['delta'] = df['fair_share_pts'] - df['baseline_pts']
    return df
