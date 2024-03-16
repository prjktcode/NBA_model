import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.static import teams
from nba_api.stats.endpoints import LeagueGameLog
from nba_api.live.nba.endpoints import scoreboard
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pickle
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Load the trained model from the file
with open('xgboost_regression_model.pkl', 'rb') as f:
    regressor = pickle.load(f)

# Get a list of all NBA teams
all_teams = teams.get_teams()

# Function to print list of teams
def print_teams(team_list):
    return pd.DataFrame(team_list)

# Query NBA scoreboard and list games
def get_games():
    board = scoreboard.ScoreBoard()
    games = board.games.get_dict()
    return games

# Specify the season you're interested in
season = '2023-24'

# Request the game logs for the specified season
game_log = LeagueGameLog(season=season)

# Get the data
game_log_data = game_log.get_data_frames()[0]

# Drop irrelevant columns
columns_to_drop = ['SEASON_ID', 'GAME_ID', 'GAME_DATE', 'VIDEO_AVAILABLE', 'MIN', 'MATCHUP', 'TEAM_NAME', 'TEAM_ABBREVIATION']
game_log_data.drop(columns=columns_to_drop, inplace=True)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode the 'WL' column
game_log_data['WL_encoded'] = label_encoder.fit_transform(game_log_data['WL'])

# Drop the original 'WL' column
game_log_data.drop(columns=['WL'], inplace=True)

# Separate features (X) and target variable (y)
X = game_log_data.drop(columns=['PTS'])
y = game_log_data['PTS']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model on the training data
regressor.fit(X_train, y_train)





# Style the title "NBA Points Prediction"
st.title("NBA Points Prediction")
st.markdown("---")  # Add a horizontal line for separation

# Style the header for "Today's NBA Games"
st.header("Today's NBA Games")
st.markdown("**Game ID**  **Home Team**  **vs.**  **Away Team**")  # Add bold text and column headers
games = get_games()

# Iterate over the games and display them with custom styling
for game in games:
    st.markdown(f"{game['gameId']}: **{game['homeTeam']['teamName']}**  vs.  **{game['awayTeam']['teamName']}**", unsafe_allow_html=True)


# Predict points for today's fixtures
predicted_points = regressor.predict(X_test)

# Display predicted points for today's fixtures
st.write("Predicted Points for Today's Fixtures:")
predictions_data = []
for idx, game in enumerate(games):
    home_team_name = game['homeTeam']['teamName']
    away_team_name = game['awayTeam']['teamName']
    predicted_score_away = predicted_points[idx * 2]
    predicted_score_home = predicted_points[idx * 2 + 1]
    predictions_data.append({
        "Home Team": home_team_name,
        "Away Team": away_team_name,
        "Predicted Points Home": predicted_score_away,
        "Predicted Points Away": predicted_score_home
    })

predictions_df = pd.DataFrame(predictions_data)
st.write(predictions_df)