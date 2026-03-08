import joblib
import pandas as pd


model = joblib.load(r'data\voting_ensemble.pkl')
print("Model loaded.")


player = {
    # ── Match context ──────────────────────────────────
    "opponent_team":                   6,   # Burnley's FPL team ID
    "was_home":                        1,   # Everton at Hill Dickinson
    "round":                          29,   # GW29, March 3 2026

    # ── This game's stats ──────────────────────────────
    "minutes":                        90,
    "starts":                          1,
    "goals_scored":                    1,   # header 32nd min
    "assists":                         0,
    "clean_sheets":                    1,   # 2-0 win
    "goals_conceded":                  0,
    "saves":                           0,
    "yellow_cards":                    0,
    "red_cards":                       0,

    # ── ICT / xG (estimates) ───────────────────────────
    "influence":                    52.0,   # scored + clean sheet DEF
    "creativity":                    5.0,
    "threat":                       18.0,
    "ict_index":                     7.5,
    "expected_goals":               0.25,   # set piece header
    "expected_assists":             0.00,
    "expected_goal_involvements":   0.25,
    "expected_goals_conceded":      0.40,   # Burnley had only 5 shots, 2 on target

    # ── Defensive stats ────────────────────────────────
    "tackles":                         2,
    "clearances_blocks_interceptions": 4,
    "recoveries":                      5,
    "defensive_contribution":          6,

    # ── Rolling averages ───────────────────────────────
    "avg_points_last3":             5.00,   # solid DEF before this
    "avg_minutes_last3":           90.00,
    "xG_last5":                     0.30,
    "xA_last5":                     0.10,
    "xGI_last5":                    0.40,
    "minutes_ratio":                1.00,

    # ── Opponent context ───────────────────────────────
    "opp_avg_goals_conceded":       2.20,   # Burnley worst defence in PL
    "opp_goals_allowed_last5":     10.00,   # leaking heavily
}
df_input = pd.DataFrame([player])
predicted_points = model.predict(df_input)[0]

print(f"\nPredicted FPL points: {predicted_points:.2f}")