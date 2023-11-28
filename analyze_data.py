import sys
import pandas as pd
import numpy as np
from scipy import stats



def main():
    playoffs = pd.read_csv(sys.argv[1])
    
    conference_finalist = playoffs[playoffs['W'] > 7]
    conference_finalist = conference_finalist[['Team', 'Season']]



if __name__ == '__main__':
    main()
