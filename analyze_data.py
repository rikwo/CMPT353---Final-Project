import sys
import pandas as pd
import numpy as np
from scipy import stats



def main():
    playoffs = pd.read_csv("combined-files\combined_playoff.csv")
    team_stats = pd.read_csv("combined-files\combined_team_stats.csv")
    
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

    #merging conference finalist to team stats to view the conference finalist team stats (i hope this makes sense)
    finalist_team_stats = pd.merge(conference_finalist, team_stats, on=['season', 'team'])

    print(finalist_team_stats)
    finalist_team_stats.to_csv('test.csv')



if __name__ == '__main__':
    main()
