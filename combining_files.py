import os
import pathlib
import sys
import pandas as pd


#use read_excel for .xlsx files
#needs to pip install openpyxl



def main():
    playoff_path = pathlib.Path(sys.argv[1])
    game_stats_path = pathlib.Path(sys.argv[2])
    team_stats_path = pathlib.Path(sys.argv[3])
    goalies_path = pathlib.Path(sys.argv[4])
    skaters_path = pathlib.Path(sys.argv[5])


    #combining all the playoff files together
    combined_playoff = pd.DataFrame()
    playoff_dir_list = os.listdir(playoff_path)
    for file in playoff_dir_list:
        file_path = os.path.join(playoff_path, file)
        df = pd.read_excel(file_path)


        combined_playoff = combined_playoff._append(df, ignore_index = True)


    #combining all the game stats files together
    combined_game_stats = pd.DataFrame()
    game_stats_dir_list = os.listdir(game_stats_path)
    for file in game_stats_dir_list:
        file_path = os.path.join(game_stats_path, file)
        df = pd.read_csv(file_path)

        df.loc[(df['GV'] > df['GH']), 'winner'] = df['Visitor']
        df.loc[(df['GH'] > df['GV']), 'winner'] = df['Home']
        df["cum_wins"] = df.groupby("winner").cumcount()+1

        combined_game_stats = combined_game_stats._append(df, ignore_index = True)


    #combining all the team stats files together
    combined_team_stats = pd.DataFrame()
    team_stats_dir_list = os.listdir(team_stats_path)
    for file in team_stats_dir_list:
        file_path = os.path.join(team_stats_path, file)
        df = pd.read_csv(file_path)


        combined_team_stats = combined_team_stats._append(df, ignore_index = True)


    #combining all the player stats files together
    players_dir_list = os.listdir(skaters_path)
    players_df = pd.DataFrame()
    for file in players_dir_list:
        file_path = os.path.join(skaters_path, file)
        df = pd.read_csv(file_path)
        players_df = players_df._append(df, ignore_index = True)


    #eliminate doubles of players who may play on powerplay or penalty kill
    players_df = players_df[players_df['situation']=='all'] 

    #Selecting only defense for their own file
    defense_df = players_df[players_df['position']=='D']
    defense_df.reset_index

    #Selecting only forwards for their own file
    forwards_df = players_df.loc[(players_df['position'] == 'L') | (players_df['position'] == 'C') | (players_df['position'] == 'R')]
    forwards_df.reset_index

    #combining all the goalie stats files together
    goalies_dir_list = os.listdir(goalies_path)
    goalies_df = pd.DataFrame()

    for file in goalies_dir_list:
        file_path = os.path.join(goalies_path, file)
        df = pd.read_csv(file_path)
        goalies_df = goalies_df._append(df, ignore_index = True)

    goalies_df.reset_index
 
    #exporting all the combined files to CSVs
    combined_playoff.to_csv('combined_playoff.csv', index=False)
    combined_game_stats.to_csv('combined_game_stats.csv', index=False)
    combined_team_stats.to_csv('combined_team_stats.csv', index=False)
    defense_df.to_csv('defense.csv', index = False)
    forwards_df.to_csv('forwards.csv', index = False)
    goalies_df.to_csv('goalies.csv', index = False)


main()

