import pandas as pd
import numpy as np

def make_matrix_cluster():
    #read in the team clusters
    team_cluster = pd.read_csv("cluster-files\Team_clusters.csv")
    forward_cluster = pd.read_csv("cluster-files\Forwards_clusters.csv")
    defense_cluster = pd.read_csv("cluster-files\defense_clusters.csv")
    goalie_cluster = pd.read_csv("cluster-files\goalie_clusters.csv")

    #fix the column names for merging
    team_cluster = team_cluster.rename(columns={"Team": "team"})
    team_cluster = team_cluster.rename(columns={"Cluster_Mode": "team_cluster"})

    teams = pd.read_csv("prepared-files\FinalTeamStats.csv")
    teams = teams.merge(team_cluster, how='left', on='team')
    teams = teams[teams['situation'] == 'all']

    #fix the column names for merging
    forward_cluster = forward_cluster.rename(columns={"Forward": "name"})
    defense_cluster = defense_cluster.rename(columns={"Defense": "name"})
    goalie_cluster = goalie_cluster.rename(columns={"Goalie": "name"})

    #read in player data
    forwards = pd.read_csv("combined-files\Forwards.csv")
    defenders = pd.read_csv("combined-files\defense.csv")
    goalies = pd.read_csv("combined-files\goalies.csv")

    #fix abbreviations
    forwards['team'] = forwards['team'].replace(['L.A', 'N.J', 'S.J', 'T.B'], ['LAK', 'NJD', 'SJS', 'TBL'])
    defenders['team'] = defenders['team'].replace(['L.A', 'N.J', 'S.J', 'T.B'], ['LAK', 'NJD', 'SJS', 'TBL'])
    goalies['team'] = goalies['team'].replace(['L.A', 'N.J', 'S.J', 'T.B'], ['LAK', 'NJD', 'SJS', 'TBL'])
         
    #add respective cluster value to each player
    forwards = forwards.merge(forward_cluster, how='left', on='name')
    defenders = defenders.merge(defense_cluster, how='left', on='name')
    goalies = goalies.merge(goalie_cluster, how='left', on='name')
    goalies = goalies[goalies['situation'] == 'all']

    #prepare teams for for loop
    abbreviations = pd.read_csv("prepared-files\Abbreviations.csv")
    team_abb = np.array(abbreviations['team'])

    #prepare cluster values for for loop
    cluster_values = teams.groupby(teams['team_cluster']).any()
    cluster_values = np.array(cluster_values.index)

    #prepare cluster values for for loop
    f_cluster_values = forwards.groupby(forwards['Cluster_Mode']).any()
    f_cluster_values = np.array(f_cluster_values.index)
    d_cluster_values = defenders.groupby(defenders['Cluster_Mode']).any()
    d_cluster_values = np.array(d_cluster_values.index)
    g_cluster_values = goalies.groupby(goalies['Cluster_Mode']).any()
    g_cluster_values = np.array(g_cluster_values.index)

    years = teams.groupby(teams['season']).any()
    years = np.array(years.index)

    #initialize dataFrame
    matrix = pd.DataFrame()

    #for loop to get desired rows of player and team clusters
    for team in team_abb:
        for year in years:
            #matrix to hold each iteration
            temp_matrix = pd.DataFrame({'team': [team], 'year' : [year]})
            #forward
            for cluster in f_cluster_values:
                #filtering by team
                team_v = forwards[forwards['team'] == team]
                #fitering by season
                team_v = team_v[team_v['season'] == year]
                #adding cluster values to respective columns, repeat for other datasets below
                temp_matrix['fwd' + str(cluster)] = team_v[team_v['Cluster_Mode'] == cluster].shape[0]
            for cluster in d_cluster_values:
                team_v = defenders[defenders['team'] == team]
                team_v = team_v[team_v['season'] == year]
                temp_matrix['def' + str(cluster)] = team_v[team_v['Cluster_Mode'] == cluster].shape[0]
            for cluster in g_cluster_values:
                team_v = goalies[goalies['team'] == team]
                team_v = team_v[team_v['season'] == year]
                temp_matrix['goalie' + str(cluster)] = team_v[team_v['Cluster_Mode'] == cluster].shape[0]
            #append the temp matrix to the main matrix
            matrix = pd.concat([matrix, temp_matrix])

    #add the team cluster values
    matrix = matrix.merge(team_cluster, how='left', on='team')

    yes_no = pd.DataFrame({'team': teams['team'], 'year': teams['season'], 'confFinal': teams['confFinal']})

    cols = matrix.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    matrix = matrix[cols]

    matrix = matrix.merge(yes_no, on=['team', 'year'])
    matrix = matrix.drop(columns='team')
    
    return matrix

def main():
    playoffs = pd.read_csv("combined-files/combined_playoff.csv")
    team_stats = pd.read_csv("combined-files/combined_team_stats.csv")
    game_stats = pd.read_csv("combined-files/combined_game_stats.csv")

    
    #changing season format to fit for teamstats ex. 20172018 -> 2017
    playoffs['Season'] = playoffs['Season'].astype('string')                                                                
    playoffs['Season'] = playoffs['Season'].str.slice_replace(4, 8, '').astype('i')
    #changing phoenix coyotes to arizona coyotes                         
    playoffs['Team'] = playoffs['Team'].replace('Phoenix Coyotes', 'Arizona Coyotes')

    #filtering for conference finalist (8 wins or higher), 2019-2020 playoffs required 11 or higher
    conference_finalist = playoffs[(playoffs['W'] > 7) & ~((playoffs['Season'] == 2019) & ~(playoffs['W'] > 10))]
    #selecting just team and season
    conference_finalist = conference_finalist[['Team', 'Season']]

    #dropping duplicate team names
    team_stats = team_stats.drop(columns=['name', 'team.1'])    
    #fixing abbreviations                                                        
    team_stats['team'] = team_stats['team'].replace(['L.A', 'N.J', 'S.J', 'T.B'], ['LAK', 'NJD', 'SJS', 'TBL'])
    #getting rid of atlanta (never made playoffs in the data)
    team_stats = team_stats[team_stats['team'] != 'ATL']

    #assigning abbreviations to each team in playoffs data
    grouped_team_playoffs = playoffs.groupby('Team').agg('count')
    grouped_team_stats = team_stats.groupby('team').agg('count')
    abbreviations = pd.DataFrame({'Team': grouped_team_playoffs.index, 'team' : grouped_team_stats.index })
    #fixing abbreviation assignment to each team
    abbreviations['team'] = abbreviations['team'].replace(['CAR', 'CBJ', 'CGY', 'CHI', 'COL', 'NJD', 'NSH', 'SEA', 'SJS', 'WPG', 'WSH'],
                                                            ['CGY', 'CAR', 'CHI', 'COL', 'CBJ', 'NSH', 'NJD', 'SJS', 'SEA', 'WSH', 'WPG'])
    
    #adding abbreviations to each conference finalist
    conference_finalist = conference_finalist.merge(abbreviations, how='left', on='Team')
    #getting ready to merge to team stats
    conference_finalist = conference_finalist.rename(columns={"Season": "season"})
    formated_finalist = pd.DataFrame({'team': conference_finalist['team'], 'season': conference_finalist['season']})

    #merging conference finalist to team stats to view the conference finalist team stats (i hope this makes sense)
    finalist_team_stats = pd.merge(formated_finalist, team_stats, on=['season', 'team'], how='left')
    finalist_team_stats['confFinal'] = 'yes'

    #separating the conference finalist from the non conference finalist
    #adapted from https://stackoverflow.com/questions/44706485/how-to-remove-rows-in-a-pandas-dataframe-if-the-same-row-exists-in-another-dataf
    not_conference = team_stats.merge(finalist_team_stats, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
    not_conference['confFinal'] = 'no'

    #concatting dataframes together and cleaning them up
    frames = [not_conference, finalist_team_stats]
    all_team_stats_final = pd.concat(frames)
    all_team_stats_final = all_team_stats_final.sort_values(by=['season', 'team'])
    all_team_stats_final = all_team_stats_final.reset_index()
    all_team_stats_final = all_team_stats_final.drop(columns='index')

    mlp_matrix = make_matrix_cluster()

    all_team_stats_final.to_csv('finalTeamStats.csv')
    abbreviations.to_csv('abbreviations.csv')
    finalist_team_stats.to_csv('confFinalsTeams.csv')
    not_conference.to_csv('not_confFinalsTeams.csv')
    mlp_matrix.to_csv('mlp_matrix.csv', header=False, index=False)




if __name__ == '__main__':
    main()
