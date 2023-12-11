import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import os
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import os

# Set the LOKY_MAX_CPU_COUNT environment variable
os.environ["LOKY_MAX_CPU_COUNT"] = "5"

# run with comment
# python team_Cluster.py


def main():
    global kmeans
    NUM_CLUST_TEAM = 12
    data = pd.read_csv("prepared-files\FinalTeamStats.csv")

    #cleaning files before making clusters
    data_NoTeam = data.drop(columns=['Unnamed: 0', 'team', 'position', 'situation', 'confFinal'])
    data_NoTeam = data_NoTeam.fillna(data_NoTeam.mean())
    data_Team = data.drop(columns=['Unnamed: 0', 'position', 'situation', 'confFinal'])
    data_Team = data_Team.fillna(data_Team)

    #settings kwargs for KMeans
    kmeans_kwargs = {
        "init": "random",
        "n_init": 12,
        "max_iter": 300,
        "random_state": 48,
    }
    # A list holds the SSE values for each k
    clusters = NUM_CLUST_TEAM
    sse = []
    for k in range(1, clusters):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(data_NoTeam)
        sse.append(kmeans.inertia_)

    #Making Plot for SSE vs Number of Clusters
    plt.plot(range(1, clusters), sse)
    plt.xticks(range(1, clusters))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.savefig("output_team_cluster")

    #Adding cluster values to dataframe
    team_cluster_assignments = kmeans.labels_
    data_Team.insert(loc=0, column='cluster', value=team_cluster_assignments)

    #Assigning cluster values to the teams
    team_list = data_Team['team'].unique()
    team_clusters = {}
    for t in team_list:
        m = data_Team.loc[data_Team['team'] == t].cluster.mode()[0]
        team_clusters[t] = m

    var = list(team_clusters.items())
    var = pd.DataFrame(var, columns=['Team', 'Cluster_Mode'])
    var.to_csv('team_clusters.csv', index=False)




if __name__ == "__main__":
    main()