
import pandas as pd
import pickle 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
credit_score_data = pd.read_csv('credit.csv')
le = LabelEncoder()
credit_score_data['Credit_Score'] = le.fit_transform(credit_score_data['Credit_Score'])
credit_score_data['Credit_Mix'] = le.fit_transform(credit_score_data['Credit_Mix'])
credit_score_data = credit_score_data.drop(['ID', 'Age', 'Customer_ID', 'Month', 'Name', 'SSN', 'Occupation', 'Type_of_Loan', 'Payment_Behaviour', 'Payment_of_Min_Amount'], axis=1)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#Select features and target
X = credit_score_data.drop('Credit_Score', axis=1)
y = credit_score_data['Credit_Score']
#split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_random_params = {
    'n_estimators': 1400,
    'min_samples_split': 2,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'max_depth': 1000,
    'criterion': 'entropy',
    'random_state' : 100,
    'verbose' : 2
}
best_random_grid = RandomForestClassifier(**best_random_params)
#Fit the model with your training data
best_random_grid.fit(X_train, y_train)
y_pred_best = best_random_grid.predict(X_test)

#Evaluate the model performance
accuracy_best = accuracy_score(y_test, y_pred_best)
print("Accuracy with best parameters : ", accuracy_best)
#Saving the model to disk
pickle.dump(best_random_grid, open('model.pkl', 'wb'))


