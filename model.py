import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
import json

warnings.filterwarnings("ignore")

SEED = 42

################################
########## DATA PREP ###########
################################

def get_rf_score(model, y_true, x_array):
    y_pred = model.predict(x_array)

    accuracy = metrics.accuracy_score(y_true, y_pred)
    f1_score = metrics.f1_score(y_true, y_pred, average='weighted')
    precision = metrics.precision_score(y_true, y_pred, average='weighted')
    recall = metrics.recall_score(y_true, y_pred, average='weighted')
    print("Accuracy =", round(accuracy, 2))
    print("F1_score =", round(f1_score, 2))
    print("Precision =", round(precision, 2))
    print("Recall =", round(recall, 2))  

    return accuracy, f1_score, precision, recall

# Load in the data
df = pd.read_csv("wine_quality.csv")

# Split into train and test sections
y = df.pop("quality")
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=SEED)

#################################
########## MODELLING ############
#################################

# build the lightgbm model
clf = RandomForestClassifier(max_depth=2, random_state=SEED).fit(x_train, y_train)

accuracy, f1_score, precision, recall = get_rf_score(clf, y_test, x_test)

# predict the results
print('train score:')
get_rf_score(clf, y_train, x_train)
print('test score:')
get_rf_score(clf, y_test, x_test)

# param_test = {
#  'class_weight': ['balanced'],
#  'n_estimators': [50, 100, 200],
#  'max_depth': [2, 5, 10]

# }
# gsearch = GridSearchCV(estimator = RandomForestClassifier(max_depth=2, n_jobs=-1, random_state=SEED), 
#                        param_grid = param_test, scoring='f1_weighted', n_jobs=-1, cv=3)
# gsearch.fit(x_train, y_train)
# gsearch.best_params_, gsearch.best_score_

# print('GridSearchCV train score:')
# get_rf_score(gsearch.best_estimator_, y_train, x_train)
# print('GridSearchCV test score:')
# accuracy, f1_score, precision, recall = get_rf_score(gsearch.best_estimator_, y_test, x_test)

# Write scores to a file
    
with open("metrics.json", 'w') as outfile:
        json.dump({ "Accuracy": accuracy, 
                    "F1_score": f1_score, 
                    "Precision": precision,
                    "Recall": recall}, outfile)
##########################################
##### PLOT FEATURE IMPORTANCE ############
##########################################

df_feature_importance = (
    pd.DataFrame({
        'feature': x_train.columns,
        # 'importance': gsearch.best_estimator_.feature_importances_,
        'importance': clf.feature_importances_,
    })
    .sort_values('importance', ascending=False)
)

feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_, 
x_train.columns)), columns=['Value', 'Feature'])

plt.figure(figsize=(12, 5))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('RandomForestClassifier Features (avg over folds)')
plt.tight_layout()
plt.show()
plt.savefig('rfc_importances.png', dpi=120)
