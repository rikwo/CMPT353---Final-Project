import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



team_stats = pd.read_csv('prepared-files\FinalTeamStats.csv')
X = team_stats.drop(columns=['confFinal', 'team', 'position', 'situation'])
y = team_stats['confFinal']

#Training random forest model 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
X_train.fillna(X_train.mean(), inplace = True)
X_test.fillna(X_train.mean(), inplace = True)
model = RandomForestClassifier(random_state = 42)
model.fit(X_train, y_train)

#scoring random forest model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')
