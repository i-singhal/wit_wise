import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from engine import calculate_fair_share

st.set_page_config(page_title="Team WISE | PUBG Fair Share Allocation", layout="wide")

st.title("PUBG Fair Share Contribution Engine (100-Point Allocator)")
st.caption("SheBuilds Datathon 2026 - Finalist Solution | Team WISE")

# Sidebar Controls
st.sidebar.header("Pillar Weights Configuration")
combat_w = st.sidebar.slider("Combat Impact Weight", 0.0, 1.0, 0.45, 0.05)
support_w = st.sidebar.slider("Support & Teamwork Weight", 0.0, 1.0, 0.35, 0.05)
tactical_w = st.sidebar.slider("Tactical & Mobility Weight", 0.0, 1.0, 0.20, 0.05)

# Normalise weights
total_w = combat_w + support_w + tactical_w
combat_w, support_w, tactical_w = combat_w / total_w, support_w / total_w, tactical_w / total_w

weights = {
    'combat_w': combat_w,
    'support_w': support_w,
    'tactical_w': tactical_w,
    'combat_sub': {'damage': 0.50, 'dbno': 0.30, 'kills': 0.20},
    'support_sub': {'revives': 0.50, 'assists': 0.35, 'heals': 0.15},
    'tactical_sub': {'survival': 0.60, 'distance': 0.40}
}

# Default Archetype Data
default_squad = pd.DataFrame([
    {"player": "P1_Fragger", "kills": 8, "damageDealt": 480, "DBNOs": 3, "assists": 0, "revives": 0, "heals": 1, "timeSurvived": 1350, "walkDistance": 1100, "rideDistance": 200},
    {"player": "P2_Support_Medic", "kills": 1, "damageDealt": 680, "DBNOs": 2, "assists": 5, "revives": 3, "heals": 7, "timeSurvived": 1500, "walkDistance": 1300, "rideDistance": 500},
    {"player": "P3_Scout_IGL", "kills": 2, "damageDealt": 310, "DBNOs": 1, "assists": 2, "revives": 1, "heals": 2, "timeSurvived": 1550, "walkDistance": 2800, "rideDistance": 1800},
    {"player": "P4_Anchor_Flex", "kills": 0, "damageDealt": 210, "DBNOs": 1, "assists": 3, "revives": 1, "heals": 2, "timeSurvived": 1280, "walkDistance": 950, "rideDistance": 300}
])

st.subheader("1. Live Match Squad Metrics")
edited_df = st.data_editor(default_squad, num_rows="fixed")

results = calculate_fair_share(edited_df, weights=weights)

# Display Key Output Metrics
st.subheader("2. 100-Point Fair Share vs Baseline Allocation")
col1, col2 = st.columns([1, 1])

with col1:
    summary_table = results[['player', 'kills', 'damageDealt', 'revives', 'baseline_pts', 'fair_share_pts', 'delta']].copy()
    summary_table['baseline_pts'] = summary_table['baseline_pts'].round(1)
    summary_table['fair_share_pts'] = summary_table['fair_share_pts'].round(1)
    summary_table['delta'] = summary_table['delta'].round(1)
    st.dataframe(summary_table.style.background_gradient(subset=['delta'], cmap='coolwarm'), use_container_width=True)

with col2:
    fig = go.Figure(data=[
        go.Bar(name='Baseline (Kills Only)', x=results['player'], y=results['baseline_pts'], marker_color='#EF553B'),
        go.Bar(name='Fair Share (Proposed)', x=results['player'], y=results['fair_share_pts'], marker_color='#00CC96')
    ])
    fig.update_layout(barmode='group', title='Point Allocation Comparison (Sum = 100)', yaxis_title="Points Allocated")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("3. Archetype Radar Breakdown")
radar_categories = ['Combat', 'Support', 'Tactical']
fig_radar = go.Figure()

for idx, row in results.iterrows():
    fig_radar.add_trace(go.Scatterpolar(
        r=[row['damageDealt']/700, (row['revives']*2 + row['assists'])/10, (row['timeSurvived']/1600 + row['totalDistance']/4000)/2],
        theta=radar_categories,
        fill='toself',
        name=row['player']
    ))
fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title="Squad Multi-Dimensional Contribution Profile")
st.plotly_chart(fig_radar, use_container_width=True)
