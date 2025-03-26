import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np
import ssl
import urllib.request
import certifi
from bs4 import BeautifulSoup
from nba_api.stats.endpoints import ScoreboardV2
from nba_api.stats.static import teams

history_df = pd.read_csv("player_24-25_stats.csv")
history_df["game_date"] = pd.to_datetime(history_df["game_date"])

TEAM_ABBR_TO_FULL = {
    "GSW": "Golden State Warriors",
    "LAL": "Los Angeles Lakers",
    "PHX": "Phoenix Suns",
    "BOS": "Boston Celtics",
    "DAL": "Dallas Mavericks",
    # Adding others as needed
}

latest_stats = (
    history_df.sort_values("game_date")
    .groupby("player_name")
    .tail(1)
    .set_index("player_name")
)

# Scrape upcoming games from NBA API
def get_upcoming_schedule():
    schedule = []
    all_teams = teams.get_teams()
    team_id_map = {team['id']: team['full_name'] for team in all_teams}

    for day_offset in [0, 1]:  # today and tomorrow
        date = (datetime.today() + timedelta(days=day_offset)).strftime('%m/%d/%Y')

        try:
            scoreboard = ScoreboardV2(game_date=date)
            games = scoreboard.get_data_frames()[0]
        except Exception as e:
            print(f"Error fetching scoreboard for {date}:", e)
            continue

        for _, row in games.iterrows():
            home_id = row['HOME_TEAM_ID']
            away_id = row['VISITOR_TEAM_ID']
            home_name = team_id_map.get(home_id, 'Unknown')
            away_name = team_id_map.get(away_id, 'Unknown')

            schedule.append({
                "game_date": row['GAME_DATE_EST'],
                "home_team": home_name,
                "away_team": away_name
            })

    return pd.DataFrame(schedule)

# Scrape team defensive ratings from BBRef
def get_team_defense_ratings():
    url = "https://www.basketball-reference.com/leagues/NBA_2024.html"
    dfs = pd.read_html(url, header=[0, 1])

    for i, df in enumerate(dfs):
        flat_cols = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
        df.columns = flat_cols

        if any("DRtg" in col for col in flat_cols) and any("Team" in col for col in flat_cols):
            team_col = next(col for col in flat_cols if "Team" in col)
            drtg_col = next(col for col in flat_cols if "DRtg" in col)

            opp_3p_col = next((col for col in flat_cols if "3P%" in col and "Defense" in col), None)
            opp_fg_col = next((col for col in flat_cols if "eFG%" in col and "Defense" in col), None)

            df = df.rename(columns={team_col: "team", drtg_col: "def_rtg"})
            df["team"] = df["team"].str.replace(r"\xa0.*", "", regex=True)

            use_cols = ["team", "def_rtg"]
            if opp_3p_col: df = df.rename(columns={opp_3p_col: "opp_3p_pct"}); use_cols.append("opp_3p_pct")
            if opp_fg_col: df = df.rename(columns={opp_fg_col: "opp_fg_pct"}); use_cols.append("opp_fg_pct")

            df = df[use_cols]
            df.set_index("team", inplace=True)
            return df

    raise ValueError("Could not find defensive rating table on BBRef.")

# Predict next game stats using rolling averages
def predict_stats(latest_row):
    return {
        "predicted_pts": latest_row.get("pts_avg_last_5", np.nan),
        "predicted_reb": latest_row.get("reb_avg_last_5", np.nan),
        "predicted_ast": latest_row.get("ast_avg_last_5", np.nan),
    }

# Build prediction rows per player
def build_predictions(latest_stats, schedule_df, defense_df):
    predictions = []

    for player, row in latest_stats.iterrows():
        abbr = row.get("team")
        player_team = TEAM_ABBR_TO_FULL.get(abbr, abbr)  # fallback to abbr if not found

        upcoming = schedule_df[(schedule_df["home_team"].str.lower() == player_team.lower()) |
                                (schedule_df["away_team"].str.lower() == player_team.lower())]

        if upcoming.empty:
            print(f"No upcoming game found for {player} ({player_team})")
            continue

        game = upcoming.iloc[0]
        opponent = game["away_team"] if game["home_team"] == player_team else game["home_team"]
        home = int(game["home_team"] == player_team)

        opp_def = defense_df.loc[opponent] if opponent in defense_df.index else {}
        pred = predict_stats(row)

        predictions.append({
            "player_name": player,
            "game_date": game["game_date"],
            "team": player_team,
            "opponent": opponent,
            "home": home,
            "injury_status": row.get("injury_status", "Unknown"),
            "opp_def_rating": opp_def.get("def_rtg", np.nan),
            "opp_3p_pct": opp_def.get("opp_3p_pct", np.nan),
            "opp_fg_pct": opp_def.get("opp_fg_pct", np.nan),
            **pred
        })

    return pd.DataFrame(predictions)

# Run Prediction Pipeline
if __name__ == "__main__":
    print("Loading upcoming schedule...")
    schedule_df = get_upcoming_schedule()

    print("Schedule DataFrame preview:")
    print(schedule_df.head())
    print("Columns:", schedule_df.columns.tolist())

    print("Loading team defense ratings...")
    defense_df = get_team_defense_ratings()

    print("Generating predictions...")
    pred_df = build_predictions(latest_stats, schedule_df, defense_df)

    pred_df.to_csv("next_game_predictions.csv", index=False)
    print("Saved predictions to next_game_predictions.csv")
