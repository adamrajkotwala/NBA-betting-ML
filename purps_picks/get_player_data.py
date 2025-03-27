import pandas as pd
import time
import numpy as np
import ssl
import urllib.request
from bs4 import BeautifulSoup
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

# Get a player's unique NBA metdata from their name
def get_player_id(full_name):
    player_list = players.get_players()
    for player in player_list:
        if player['full_name'] == full_name:
            return player['id']
    return None

# Download player game logs from nba_api
def get_player_logs(player_id, season='2024'):
    # This gets all regular season game logs for a player
    gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    df = gamelog.get_data_frames()[0]
    return df

# Compute rolling averages over the last 5 games
def add_rolling_averages(df, player_name, stat_columns, window=5):
    df = df.copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='%b %d, %Y')
    df = df.sort_values(by=['PLAYER_NAME', 'GAME_DATE']).reset_index(drop=True)

    for stat in stat_columns:
        avg_col = f'{stat}_avg_last_{window}'
        df[avg_col] = np.nan

        # Group and assign
        for name, group in df.groupby('PLAYER_NAME'):
            group = group.sort_values('GAME_DATE')
            rolling = group[stat].shift(1).rolling(window).mean().reset_index(drop=True)

            # Align back to original DataFrame
            df.loc[group.index, avg_col] = rolling

    return df


# Add features like HOME, OPPONENT, REST DAYS
def add_game_features(df):
    # HOME: 1 if the team played at home, 0 if away
    df['HOME'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)

    # OPPONENT: team abbreviation from the MATCHUP string
    df['OPPONENT'] = df['MATCHUP'].apply(lambda x: x.split()[-1])

    # Convert GAME_DATE to datetime
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='%b %d, %Y')

    # Calculate rest days between games
    df = df.sort_values(by=['PLAYER_NAME', 'GAME_DATE'])
    df['PREV_GAME_DATE'] = df.groupby('PLAYER_NAME')['GAME_DATE'].shift(1)
    df['DAYS_REST'] = (df['GAME_DATE'] - df['PREV_GAME_DATE']).dt.days

    return df

# Full pipeline to build a dataset for one player
def build_player_dataset(player_name, season='2024', team_defense_ratings=np.nan):
    player_id = get_player_id(player_name)
    if player_id is None or pd.isna(player_id):
        print(f"Player '{player_name}' not found.")
        return pd.DataFrame()

    df = get_player_logs(player_id, season)

    df['PLAYER_NAME'] = player_name

    # Infer team from game logs (use 'MATCHUP' column)
    df['TEAM'] = df['MATCHUP'].apply(lambda x: x.split()[0] if isinstance(x, str) else 'Unknown')

    # Add engineered features
    df = add_rolling_averages(df, player_name, stat_columns=['PTS', 'REB', 'AST'], window=5)
    df = add_game_features(df)
    return df

def get_current_injuries():
    url = 'https://www.espn.com/nba/injuries'
    try:
        context = ssl._create_unverified_context()
        html = urllib.request.urlopen(url, context=context).read()
        soup = BeautifulSoup(html, 'html.parser')
        tables = pd.read_html(html)
    except Exception as e:
        print("Error reading ESPN injury page:", e)
        return pd.DataFrame()

    injury_tables = []
    for table in tables:
        if {'NAME', 'STATUS'}.issubset(table.columns):
            table = table.copy()
            injury_tables.append(table)

    if not injury_tables:
        print("No valid injury tables found on ESPN.")
        return pd.DataFrame()

    injury_df = pd.concat(injury_tables, ignore_index=True)
    injury_df.rename(columns={'NAME': 'PLAYER_NAME', 'STATUS': 'INJURY_STATUS'}, inplace=True)

    return injury_df[['PLAYER_NAME', 'INJURY_STATUS']]

def add_injury_status(df, injury_df):
    if injury_df.empty or 'PLAYER_NAME' not in injury_df.columns:
        df['INJURY_STATUS'] = 'Unknown'
        return df

    df = df.copy()
    injury_df['PLAYER_NAME'] = injury_df['PLAYER_NAME'].str.lower()
    df['PLAYER_NAME_LOWER'] = df['PLAYER_NAME'].str.lower()

    # Merge on player name
    df = df.merge(injury_df, how='left', left_on='PLAYER_NAME_LOWER', right_on='PLAYER_NAME')
    df.drop(columns=['PLAYER_NAME_LOWER', 'PLAYER_NAME_y'], inplace=True)
    df.rename(columns={'PLAYER_NAME_x': 'PLAYER_NAME'}, inplace=True)

    df['INJURY_STATUS'] = df['INJURY_STATUS'].fillna('Active')
    return df

def build_multi_player_dataset(player_names, season='2024', team_defense_ratings=np.nan):
    all_data = []

    for name in player_names:
        print(f"Processing {name}...")
        df = build_player_dataset(name, season, team_defense_ratings)
        if not df.empty:
            all_data.append(df)
        time.sleep(1)  # Sleep to avoid hitting NBA API rate limits

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    else:
        print("No data collected.")
        return pd.DataFrame()

def save_dataset(df, filename):
    df.to_csv(filename, index=False)
    print(f"Saved dataset to {filename}")

def clean_column_names(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
        .str.replace('[^a-z0-9_]', '', regex=True)
    )
    return df

def reorder_columns(df):
    preferred_order = ['season_id', 'player_id', 'player_name', 'team', 'game_date']
    actual_order = [col for col in preferred_order if col in df.columns]
    rest = [col for col in df.columns if col not in actual_order]
    return df[actual_order + rest]

def main():
    player_list = ['LeBron James', 'Stephen Curry', 'Kevin Durant', 'Jayson Tatum', 'Luka Dončić', 'Miles Bridges', 'Karl-Anthony Towns', 'Victor Wembanyama']
    season = '2024'

    print("Scraping current injury data from ESPN...")
    injury_data = get_current_injuries()

    all_data = []

    for player_name in player_list:
        print(f"Processing {player_name}...")

        df = build_player_dataset(player_name, season=season, team_defense_ratings=None)

        if df.empty:
            print(f"Skipping {player_name} due to missing or incomplete data.")
            continue

        df = add_injury_status(df, injury_data)

        all_data.append(df)
        time.sleep(1)  # To avoid NBA API rate limiting

    # Combine all players' data into one DataFrame
    if all_data:
        multi_df = pd.concat(all_data, ignore_index=True)

        multi_df = clean_column_names(multi_df)
        multi_df = reorder_columns(multi_df)

        save_dataset(multi_df, 'player_24-25_stats.csv')
    else:
        print("No player data was collected.")


if __name__ == '__main__':
    main()
