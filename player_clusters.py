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
    NUM_FORWARD_CLUSTERS = 10
    NUM_DEFENSE_CLUSTERS = 7
    NUM_GOALIE_CLUSTERS = 6
    forwards = pd.read_csv("combined-files/forwards.csv")
    defense = pd.read_csv("combined-files/defense.csv")
    goalies = pd.read_csv("combined-files/goalies.csv")

    scaler = MinMaxScaler()

    data_no_name_forwards = forwards.drop(columns=['name', 'team', 'position', 'situation'])
    data_no_name_forwards = scaler.fit_transform(data_no_name_forwards)
    data_forwards = forwards.drop(columns=['team', 'position', 'situation'])
    data_forwards = data_forwards.fillna(data_forwards)

    data_no_name_defense = defense.drop(columns=['name', 'team', 'position', 'situation'])
    data_no_name_defense = scaler.fit_transform(data_no_name_defense)
    data_defense = defense.drop(columns=['team', 'position', 'situation'])
    data_defense = data_defense.fillna(data_defense)

    data_no_name_goalies = goalies.drop(columns=['name', 'team', 'position', 'situation'])
    data_no_name_goalies = scaler.fit_transform(data_no_name_goalies)
    data_goalies = goalies.drop(columns=['team', 'position', 'situation'])
    data_goalies = data_goalies.fillna(data_goalies)

    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
    }
    #Beginning of creating forward clusters
    # A list holds the SSE values for each k
    clusters = NUM_FORWARD_CLUSTERS
    sse_forwards = []
    for k in range(1, clusters):
        kmeans_forwards = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans_forwards.fit(data_no_name_forwards)
        sse_forwards.append(kmeans_forwards.inertia_)

    plt.plot(range(1, clusters), sse_forwards)
    plt.xticks(range(1, clusters))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.savefig("output_forward_cluster")

    forward_cluster_assignments = kmeans_forwards.labels_
    data_forwards.insert(loc=0, column='cluster', value=forward_cluster_assignments)

    forward_list = data_forwards['name'].unique()
    forward_clusters = {}
    for forward in forward_list:
        player = data_forwards.loc[data_forwards['name'] == forward].cluster.mode()[0]
        forward_clusters[forward] = player

    var = list(forward_clusters.items())
    var = pd.DataFrame(var, columns=['Forward', 'Cluster_Mode'])
    var.to_csv('forwards_clusters.csv', index=False)


    #Beginning of creating defense clusters
    clusters = NUM_DEFENSE_CLUSTERS
    sse_defense = []
    for k in range(1, clusters):
        kmeans_defense = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans_defense.fit(data_no_name_defense)
        sse_defense.append(kmeans_defense.inertia_)

    plt.plot(range(1, clusters), sse_defense)
    plt.xticks(range(1, clusters))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.savefig("output_defense_cluster")

    defense_cluster_assignments = kmeans_defense.labels_
    data_defense.insert(loc=0, column='cluster', value=defense_cluster_assignments)

    defense_list = data_defense['name'].unique()
    defense_clusters = {}
    for d in defense_list:
        player = data_defense.loc[data_defense['name'] == d].cluster.mode()[0]
        defense_clusters[d] = player

    var = list(defense_clusters.items())
    var = pd.DataFrame(var, columns=['Defense', 'Cluster_Mode'])
    var.to_csv('defense_clusters.csv', index=False)

    #different kwargs for goalies, as it was only making 2 groups with the other kwargs
    goalie_kmeans_kwargs = {
        "init": "random",
        "n_init": 'auto',
        "max_iter": 300,
        "random_state": 48,
    }
    #Beginning of creating goalie clusters
    clusters = NUM_GOALIE_CLUSTERS
    sse_goalies = []
    for k in range(1, clusters):
        kmeans_goalies = KMeans(n_clusters=k, **goalie_kmeans_kwargs)
        kmeans_goalies.fit(data_no_name_goalies)
        sse_goalies.append(kmeans_goalies.inertia_)

    plt.plot(range(1, clusters), sse_goalies)
    plt.xticks(range(1, clusters))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.savefig("output_goalie_cluster")

    goalie_cluster_assignments = kmeans_goalies.labels_
    data_goalies.insert(loc=0, column='cluster', value=goalie_cluster_assignments)

    goalie_list = data_goalies['name'].unique()
    goalie_clusters = {}
    for goalie in goalie_list:
        player = data_goalies.loc[data_goalies['name'] == goalie].cluster.mode()[0]
        goalie_clusters[goalie] = player

    var = list(goalie_clusters.items())
    var = pd.DataFrame(var, columns=['Goalie', 'Cluster_Mode'])
    var.to_csv('goalie_clusters.csv', index=False)



if __name__ == "__main__":
    main()