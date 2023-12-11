# CMPT353---Final-Project

This data pipeline uses NHL player and team data to train a model and predict the chance of each team has to make the playoff conference finals.

## Requried Libraries
Run pip install `<library>`
- numpy
- pandas
- openpxyl
- sklearn
- torch
- tqdm
- matplotlib


## Order of Execution

- Using KNN to predict the outcome of conference final, 86% accuracy
- in terminal
```bash
  python playoff_predict
```
- Using neural net to predict the outcome of conference final, 96% accuracy
- in terminal
```bash
  python NHL_model_evaluations.py prepared-files\mlp_matrix.csv
```

Begin by running the combining files with 'python combining_files.py data-files\playoff-results data-files\game-stats data-files\team-stats data-files\goalie-stats data-files\player-stats output'

## Running the Preparation and Cleaning Files
- if you would like to run `analyze_data.py`, run `python analyze_data.py` in the terminal
- if you would like to run `combining_files.py`, run `combining_files.py data-files\playoff-results data-files\game-stats data-files\team-stats data-files\goalie-stats data-files\player-stats` in the terminal
- if you would like to run `player_clusters.py`, run `python player_clusters.py` in the terminal
- if you would like to run `team_Cluster.py`, run `python team_Cluster.py` in the terminal


