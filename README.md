# PUBG Fair Share Contribution System (100-Point Reward Engine)
**Datathon Track**: SheBuilds Datathon 2026 | Finalist Round 2
**Mentor**: Ms. Gwendalynn Lim
**Team**: Team WISE

## 1. Overview & Objective
Current competitive Battle Royale point allocations disproportionately reward final-bullet fraggers while penalizing team-enabling roles (scouts, medics, anchors). This project implements an axiomatic, transparent 100-point reward allocation system that captures multi-dimensional team contributions.

## 2. Mathematical Framework
- **Pillar 1: Combat Impact (45%)** = 0.50 * Damage + 0.30 * DBNOs (Knockdowns) + 0.20 * Kills
- **Pillar 2: Support & Teamwork (35%)** = 0.50 * Revives + 0.35 * Assists + 0.15 * Capped Heals
- **Pillar 3: Tactical & Mobility (20%)** = 0.60 * Survival Time + 0.40 * Total Distance Traveled
- **Normalization**: Intra-squad sum normalization guarantees $\sum P_i = 100$ per squad without divide-by-zero errors.

## 3. Quickstart
```bash
pip install -r requirements.txt
python test_run.py       # Run CLI evaluation
streamlit run app.py     # Launch interactive dashboard
```
