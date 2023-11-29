import sys
import pandas as pd
import numpy as np
from scipy import stats



def main():
    playoffs = pd.read_csv("combined-files\combined_playoff.csv")
    team_stats = pd.read_csv("combined-files\combined_team_stats.csv")
    
    conference_finalist = playoffs[(playoffs['W'] > 7) & ~(playoffs['Season'] == 20192020)]
    conference_finalist = conference_finalist[['Team', 'Season']]
    conference_finalist['Season'] = conference_finalist['Season'].astype('string')
    conference_finalist['Season'] = conference_finalist['Season'].str.slice_replace(0, 4, '').astype('i')

    print(conference_finalist.dtypes)



if __name__ == '__main__':
    main()
