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

TEAM_FULL_TO_ABBR = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BRK",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS"
}

TEAM_ABBR_TO_FULL = {abbr: full for full, abbr in TEAM_FULL_TO_ABBR.items()}

latest_stats = (
    history_df.sort_values("game_date")
    .groupby("player_name")
    .tail(1)
    .set_index("player_name")
)

# Scrape upcoming games from NBA API
def get_upcoming_schedule(days_ahead=7):
    schedule = []
    all_teams = teams.get_teams()
    team_id_map = {team['id']: team['full_name'] for team in all_teams}

    for day_offset in range(days_ahead):
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
                "game_date": pd.to_datetime(row['GAME_DATE_EST']),
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

            df = df.rename(columns={team_col: "team", drtg_col: "def_rtg"})
            df["team"] = df["team"].str.replace(r"\xa0.*", "", regex=True)

            use_cols = ["team", "def_rtg"]

            df = df[use_cols]
            df.set_index("team", inplace=True)
            df.index = df.index.str.replace("*", "", regex=False).str.strip()
            return df

    raise ValueError("Could not find defensive rating table on BBRef.")

def get_team_srs_ratings():
    url = "https://www.basketball-reference.com/leagues/NBA_2024.html"
    dfs = pd.read_html(url, header=[0, 1])  # Handle multi-level headers

    for df in dfs:
        # Flatten multi-index column names
        flat_cols = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
        df.columns = flat_cols

        if any("SRS" in col for col in flat_cols) and any("Team" in col for col in flat_cols):
            team_col = next(col for col in flat_cols if "Team" in col)
            srs_col = next(col for col in flat_cols if "SRS" in col)

            df = df.rename(columns={team_col: "team", srs_col: "srs"})
            df["team"] = df["team"].str.replace(r"\xa0.*", "", regex=True)
            df["team"] = df["team"].str.replace("*", "", regex=False).str.strip()

            df = df[["team", "srs"]]
            df.set_index("team", inplace=True)
            return df["srs"]

    raise ValueError("Could not find SRS table.")

def compute_opponent_averages(history_df, player_name, opponent_name):
    abbr = TEAM_FULL_TO_ABBR.get(opponent_name)
    if not abbr:
        return {
            "avg_pts_vs_opp": np.nan,
            "avg_reb_vs_opp": np.nan,
            "avg_ast_vs_opp": np.nan,
            "avg_stl_vs_opp": np.nan,
            "avg_blk_vs_opp": np.nan,
            "avg_3pm_vs_opp": np.nan,
            "avg_3pa_vs_opp": np.nan,
        }

    player_games = history_df[history_df["player_name"] == player_name]
    relevant_games = player_games[player_games["opponent"] == abbr]

    if relevant_games.empty:
        return {
            "avg_pts_vs_opp": np.nan,
            "avg_reb_vs_opp": np.nan,
            "avg_ast_vs_opp": np.nan,
            "avg_stl_vs_opp": np.nan,
            "avg_blk_vs_opp": np.nan,
            "avg_3pm_vs_opp": np.nan,
            "avg_3pa_vs_opp": np.nan,
        }

    return {
        "avg_pts_vs_opp": relevant_games["pts"].mean(),
        "avg_reb_vs_opp": relevant_games["reb"].mean(),
        "avg_ast_vs_opp": relevant_games["ast"].mean(),
        "avg_stl_vs_opp": relevant_games["stl"].mean() if "stl" in relevant_games else np.nan,
        "avg_blk_vs_opp": relevant_games["blk"].mean() if "blk" in relevant_games else np.nan,
        "avg_3pm_vs_opp": relevant_games["3pm"].mean() if "3pm" in relevant_games else np.nan,
        "avg_3pa_vs_opp": relevant_games["3pa"].mean() if "3pa" in relevant_games else np.nan,
    }


# Predict next game stats using rolling averages
def predict_stats(latest_row):
    return {
        "predicted_pts": latest_row.get("pts_avg_last_5", np.nan),
        "predicted_reb": latest_row.get("reb_avg_last_5", np.nan),
        "predicted_ast": latest_row.get("ast_avg_last_5", np.nan),
    }

# Build prediction rows per player
def build_predictions(latest_stats, schedule_df, defense_df, srs_df):
    predictions = []
    now = pd.Timestamp.now()

    for player, row in latest_stats.iterrows():
        abbr = row.get("team")
        injury_status = row.get("injury_status", "").lower()
        if "out" in injury_status:
            print(f"Skipping {player} (marked as OUT)")
            continue

        player_team = TEAM_ABBR_TO_FULL.get(abbr, abbr)

        # Filter to future games only
        schedule_df["game_date"] = pd.to_datetime(schedule_df["game_date"])
        future_games = schedule_df[
            ((schedule_df["home_team"].str.lower() == player_team.lower()) |
             (schedule_df["away_team"].str.lower() == player_team.lower())) &
            (schedule_df["game_date"] > now)
        ]

        if future_games.empty:
            print(f"No upcoming game found for {player} ({player_team})")
            continue

        game = future_games.iloc[0]
        opponent = game["away_team"] if game["home_team"] == player_team else game["home_team"]
        home = int(game["home_team"] == player_team)

        # Get abbreviation version of opponent to match historical data
        opponent_clean = opponent.replace("*", "").strip()
        opponent_abbr = TEAM_FULL_TO_ABBR.get(opponent_clean)

        if not opponent_abbr:
            print(f"Abbreviation not found for opponent: {opponent_clean}")
            continue

        # Filter for past games vs opponent
        past_vs_opp = history_df[
            (history_df["player_name"] == player) &
            (history_df["team"] == abbr) &
            (history_df["opponent"] == opponent_abbr)
        ]

        avg_pts_vs_opp = past_vs_opp["pts"].mean() if not past_vs_opp.empty else np.nan
        avg_reb_vs_opp = past_vs_opp["reb"].mean() if not past_vs_opp.empty else np.nan
        avg_ast_vs_opp = past_vs_opp["ast"].mean() if not past_vs_opp.empty else np.nan
        avg_3pm_vs_opp = past_vs_opp["fg3m"].mean() if "fg3m" in past_vs_opp else np.nan
        avg_3pa_vs_opp = past_vs_opp["fg3a"].mean() if "fg3a" in past_vs_opp else np.nan
        avg_stl_vs_opp = past_vs_opp["stl"].mean() if "stl" in past_vs_opp else np.nan
        avg_blk_vs_opp = past_vs_opp["blk"].mean() if "blk" in past_vs_opp else np.nan

        # Defensive rating
        opp_def = defense_df.loc[opponent_clean] if opponent_clean in defense_df.index else {}

        # Blowout risk
        team_srs = srs_df.get(player_team, np.nan)
        opp_srs = srs_df.get(opponent_clean, np.nan)
        blowout_risk = abs(team_srs - opp_srs) if not np.isnan(team_srs) and not np.isnan(opp_srs) else np.nan

        # Rolling average predictions
        pred = predict_stats(row)

        predictions.append({
            "player_name": player,
            "game_date": game["game_date"],
            "team": player_team,
            "opponent": opponent,
            "home": home,
            "injury_status": row.get("injury_status", "Unknown"),
            "opp_def_rating": opp_def.get("def_rtg", np.nan),
            "blowout_risk": blowout_risk,
            "avg_pts_vs_opp": avg_pts_vs_opp,
            "avg_reb_vs_opp": avg_reb_vs_opp,
            "avg_ast_vs_opp": avg_ast_vs_opp,
            "avg_3pm_vs_opp": avg_3pm_vs_opp,
            "avg_3pa_vs_opp": avg_3pa_vs_opp,
            "avg_stl_vs_opp": avg_stl_vs_opp,
            "avg_blk_vs_opp": avg_blk_vs_opp,
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

    print("Loading team SRS ratings...")
    srs_ratings = get_team_srs_ratings()

    print("Generating predictions...")
    pred_df = build_predictions(latest_stats, schedule_df, defense_df, srs_ratings)

    pred_df.to_csv("next_game_predictions.csv", index=False)
    print("Saved predictions to next_game_predictions.csv")
