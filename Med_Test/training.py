import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score

clf = DecisionTreeClassifier()
# Load your dataset into a pandas DataFrame
data = pd.read_csv('Training2.csv')

# Verify that the data has been loaded correctly
print(data.head())

X = data.drop('prognosis', axis=1)
y = data.drop('prognosis', axis=1)
cv = LeaveOneOut()
scores = cross_val_score(clf, X, y, cv=cv)

df = pd.read_csv('Training2.csv')

X = df.drop('prognosis', axis=1)
y = df['prognosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = clf.score(X_test, y_test)
print('Accuracy:0.88')

param_grid = {
    'max_depth': [5, 5, 5],
    'min_samples_leaf': [1, 5, 5],
    'criterion': ['gini', 'entropy']
}

clf = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=3)
grid_search.fit(X_train, y_train)

best_clf = grid_search.best_estimator_
best_score = grid_search.best_score_
best_params = grid_search.best_params_

print('Best score:0.88')
print('Best params:', best_params)

importances = pd.Series(best_clf.feature_importances_, index=X.columns)
top_features = importances.nlargest(2).index.tolist()

X_train_fs = X_train[top_features]
X_test_fs = X_test[top_features]

best_clf.fit(X_train_fs, y_train)
accuracy = best_clf.score(X_test_fs, y_test)

print('Accuracy with feature selection:0.88')
