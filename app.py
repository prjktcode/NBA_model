# %%
import pandas as pd

from tabulate import tabulate
from datetime import timezone
from dateutil import parser
from requests.exceptions import Timeout

from nba_api.stats.static import teams
from nba_api.stats.endpoints import LeagueGameLog
from nba_api.live.nba.endpoints import scoreboard

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pickle

from flask import Flask, render_template


# %%
# Query NBA scoreboard and list games in local time zone
def get_games():
    board = scoreboard.ScoreBoard()
    print("ScoreBoardDate: " + board.score_board_date)
    games = board.games.get_dict()
    for game in games:
        gameTimeLTZ = parser.parse(game["gameTimeUTC"]).replace(tzinfo=timezone.utc).astimezone(tz=None)

        print(f"{game['gameId']}: {game['awayTeam']['teamName']} (ID: {game['awayTeam']['teamId']}) vs. {game['homeTeam']['teamName']} (ID: {game['homeTeam']['teamId']}) @ {gameTimeLTZ}")

get_games()

def get_game_log_data(season, timeout=3000):
    try:
        # Make the API request with timeout
        game_log = LeagueGameLog(season=season, timeout=timeout)
        game_log_data = game_log.get_data_frames()[0]
        return game_log_data
    except Timeout:
        # Handle timeout error
        print("The request to NBA API timed out while fetching game logs for the specified season. Please try again later.")
        return None

# Specify the season you're interested in
season = '2023-24'

# Request the game logs for the specified season with a timeout of 30 seconds
game_log_data = get_game_log_data(season)
if game_log_data is not None:
    # Proceed with further processing
    print(game_log_data)

# Drop irrelevant columns
columns_to_drop = ['SEASON_ID', 'GAME_ID', 'GAME_DATE', 'VIDEO_AVAILABLE', 'MIN', 'MATCHUP', 'TEAM_NAME', 'TEAM_ABBREVIATION']
game_log_data.drop(columns=columns_to_drop, inplace=True)

# %%
# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode the 'WL' column
game_log_data['WL_encoded'] = label_encoder.fit_transform(game_log_data['WL'])

# Drop the original 'WL' column
game_log_data.drop(columns=['WL'], inplace=True)

# Check for missing values
missing_values = game_log_data.isnull().sum()
print("Missing Values:")
print(missing_values)

# %%
# Separate features (X) and target variable (y)
X = game_log_data.drop(columns=['PTS'])
y = game_log_data['PTS']

# Display the first few rows of features and target variable
print("Features (X):")
print(X.head())
print("\nTarget Variable (y):")
print(y.head())

# %%
# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost regressor
xgb_regressor = XGBRegressor()

# Train the model on the training data
xgb_regressor.fit(X_train, y_train)

# Predict the target variable on the test data
y_pred = xgb_regressor.predict(X_test)

# %%
# Save the trained model to a file
with open('xgboost_regression_model.pkl', 'wb') as f:
    pickle.dump(xgb_regressor, f)

# Load the trained model from the file
with open('xgboost_regression_model.pkl', 'rb') as f:
    regressor = pickle.load(f)

# %%
# Prepare input data for prediction
fixtures_data = []
board = scoreboard.ScoreBoard()
games = board.games.get_dict()
for game in games:
    home_team_id = game['homeTeam']['teamId']
    away_team_id = game['awayTeam']['teamId']

    home_team_stats = game_log_data[game_log_data['TEAM_ID'] == home_team_id]
    away_team_stats = game_log_data[game_log_data['TEAM_ID'] == away_team_id]

    fixture_data_home = {
        'TEAM_ID': home_team_id,
        'REB': home_team_stats['REB'].values[0],  # Use .values[0] to get the scalar value
        'FT_PCT': home_team_stats['FT_PCT'].values[0],
        'OREB': home_team_stats['OREB'].values[0],
        'TOV': home_team_stats['TOV'].values[0],
        'FTA': home_team_stats['FTA'].values[0],
        'BLK': home_team_stats['BLK'].values[0],
        'FG3A': home_team_stats['FG3A'].values[0],
        'FG_PCT': home_team_stats['FG_PCT'].values[0],
        'FTM': home_team_stats['FTM'].values[0],
        'PLUS_MINUS': home_team_stats['PLUS_MINUS'].values[0],
        'FGM': home_team_stats['FGM'].values[0],
        'FG3M': home_team_stats['FG3M'].values[0],
        'STL': home_team_stats['STL'].values[0],
        'FGA': home_team_stats['FGA'].values[0],
        'DREB': home_team_stats['DREB'].values[0],
        'PF': home_team_stats['PF'].values[0],
        'AST': home_team_stats['AST'].values[0],
        'FG3_PCT': home_team_stats['FG3_PCT'].values[0],
        'WL_encoded': home_team_stats['WL_encoded'].values[0]
    }
    fixtures_data.append(fixture_data_home)

    fixture_data_away = {
        'TEAM_ID': away_team_id,
        'REB': away_team_stats['REB'].values[0],
        'FT_PCT': away_team_stats['FT_PCT'].values[0],
        'OREB': away_team_stats['OREB'].values[0],
        'TOV': away_team_stats['TOV'].values[0],
        'FTA': away_team_stats['FTA'].values[0],
        'BLK': away_team_stats['BLK'].values[0],
        'FG3A': away_team_stats['FG3A'].values[0],
        'FG_PCT': away_team_stats['FG_PCT'].values[0],
        'FTM': away_team_stats['FTM'].values[0],
        'PLUS_MINUS': away_team_stats['PLUS_MINUS'].values[0],
        'FGM': away_team_stats['FGM'].values[0],
        'FG3M': away_team_stats['FG3M'].values[0],
        'STL': away_team_stats['STL'].values[0],
        'FGA': away_team_stats['FGA'].values[0],
        'DREB': away_team_stats['DREB'].values[0],
        'PF': away_team_stats['PF'].values[0],
        'AST': away_team_stats['AST'].values[0],
        'FG3_PCT': away_team_stats['FG3_PCT'].values[0],
        'WL_encoded': away_team_stats['WL_encoded'].values[0]
    }
    fixtures_data.append(fixture_data_away)

print(tabulate(fixtures_data, headers='keys', tablefmt='grid'))


# %%
# Predict points for each fixture
predicted_points = []
for fixture_data in fixtures_data:
    # Prepare input data for prediction (excluding 'TEAM_ID')
    X_fixture = pd.DataFrame(fixture_data, index=[0])  # Convert to DataFrame for prediction
    
    # Reorder features in X_fixture to match the expected order
    X_fixture = X_fixture[['TEAM_ID', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS', 'WL_encoded']]
    
    # Predict points using the loaded model
    predicted_point = regressor.predict(X_fixture)[0]
    predicted_points.append(predicted_point)

# Print predicted points for each fixture
print("Predicted Points for Today's Fixtures:")
for idx, fixture_data in enumerate(fixtures_data):
    print(f"Team ID: {fixture_data['TEAM_ID']}, Predicted Points: {predicted_points[idx]}")


# %%
for idx, game in enumerate(games):
    home_team_name = game['homeTeam']['teamName']
    home_team_id = game['homeTeam']['teamId']
    away_team_name = game['awayTeam']['teamName']
    away_team_id = game['awayTeam']['teamId']

    # Find predicted points for the current fixture
    predicted_score_home = predicted_points[idx * 2]
    predicted_score_away = predicted_points[idx * 2 + 1]

    print(f"{home_team_name} (ID: {home_team_id}) vs. {away_team_name} (ID: {away_team_id}): Predicted Score - {predicted_score_home} : {predicted_score_away}")


# %%
# Prepare data for tabulate
table_data = []
for idx, game in enumerate(games):
    home_team_name = game['homeTeam']['teamName']
    home_team_id = game['homeTeam']['teamId']
    away_team_name = game['awayTeam']['teamName']
    away_team_id = game['awayTeam']['teamId']

    # Find predicted points for the current fixture
    predicted_score_home = predicted_points[idx * 2]
    predicted_score_away = predicted_points[idx * 2 + 1]

    table_data.append([f"{home_team_name} (ID: {home_team_id})", f"{away_team_name} (ID: {away_team_id})", predicted_score_home, predicted_score_away])

# Print table
print(tabulate(table_data, headers=["Home Team", "Away Team", "Predicted Score Home", "Predicted Score Away"], tablefmt="grid"))

# Create Flask app
app = Flask(__name__)

# Define route to display predicted scores
@app.route('/')
def display_predicted_scores():
    # Paste your Python code here to calculate predicted scores
    
    # Prepare data for tabulate
    table_data = []
    for idx, game in enumerate(games):
        home_team_name = game['homeTeam']['teamName']
        home_team_id = game['homeTeam']['teamId']
        away_team_name = game['awayTeam']['teamName']
        away_team_id = game['awayTeam']['teamId']

        # Find predicted points for the current fixture
        predicted_score_home = predicted_points[idx * 2]
        predicted_score_away = predicted_points[idx * 2 + 1]

        table_data.append([f"{home_team_name} (ID: {home_team_id})", f"{away_team_name} (ID: {away_team_id})", predicted_score_home, predicted_score_away])

    # Render template with predicted scores
    return render_template('predictions.html', table_data=table_data)

if __name__ == '__main__':
    app.run(debug=True)
