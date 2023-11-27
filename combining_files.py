import os
import pathlib
import sys
import numpy as np
import pandas as pd


#use read_excel for .xlsx files




def main():
    playoff_path = pathlib.Path(sys.argv[1])
    game_stats_path = pathlib.Path(sys.argv[2])
    team_stats_path = pathlib.Path(sys.argv[3])
    output_directory = pathlib.Path(sys.argv[4])


    combined_playoff = pd.DataFrame()


    playoff_dir_list = os.listdir(playoff_path)
    for file in playoff_dir_list:
        file_path = os.path.join(playoff_path, file)
        df = pd.read_excel(file_path)


        combined_playoff = combined_playoff._append(df, ignore_index = True)


    combined_game_stats = pd.DataFrame()


    game_stats_dir_list = os.listdir(game_stats_path)
    for file in game_stats_dir_list:
        file_path = os.path.join(game_stats_path, file)
        df = pd.read_csv(file_path, delimiter='\t')


        combined_game_stats = combined_game_stats._append(df, ignore_index = True)


    combined_team_stats = pd.DataFrame()


    team_stats_dir_list = os.listdir(team_stats_path)
    for file in team_stats_dir_list:
        file_path = os.path.join(team_stats_path, file)
        df = pd.read_csv(file_path, delimiter='\t')


        combined_team_stats = combined_team_stats._append(df, ignore_index = True)

    combined_playoff.to_csv('combined_playoff.csv', index=False)
    combined_game_stats.to_csv('combined_game_stats.csv', index=False)
    combined_team_stats.to_csv('combined_team_stats.csv', index=False)


main()

