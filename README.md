**NBA Game Score Prediction with XGBoost**
This repository contains Python code for an NBA game score prediction model using XGBoost and a Flask application to display the predictions. The program uses nba_api to get data and fixtures from Stats.NBA

**Functionality**
**Data Acquisition and Processing:**

Retrieves a list of NBA teams.

Queries the live NBA scoreboard for games.

Fetches historical game logs for a specified season.

Cleans and prepares the data for modeling.

Feature Engineering and Model Training:

Extracts features from game logs relevant to predicting points scored.

Trains an XGBoost regression model to predict game scores.

Evaluates the model's performance using various metrics.


**Prediction and Flask App:**

**Loads the trained XGBoost model.**
Predicts points for upcoming games based on their team's historical stats.
Provides a Flask application to display the predicted scores in a user-friendly format.


**How to Use**
**Prerequisites:**

Python 3.x with libraries like pandas, numpy, XGBoost, scikit-learn, requests, tabulate, and Flask installed.
nba_api
Steps:

**Clone the repository.**
Run the script: python nba_game_prediction.py
This will print the predicted scores for today's games on the console.

**Web App:**
Navigate to the project directory in your terminal.
Run the Flask app: python nba_game_prediction.py --run-app
Access the predictions at http://127.0.0.1:5000/ in your web browser.
Note: Depending on your environment, you may need to adjust the Flask app execution command.

**Dependencies**
nba_api
pandas
numpy
XGBoost
scikit-learn
requests
tabulate
Flask
File Structure
