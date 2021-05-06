from sklearn.preprocessing import LabelEncoder
import statistics as stats
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

df = pd.read_csv("D:\\Limudim\\train_Loan.csv")

# Print all missing values for evaluation
print(df.apply(lambda x: sum(x.isnull()),axis=0)) # This line print the amount of missing value in each column

# Fill missing values in some attribute :
df['Gender'].fillna(stats.mode(df['Gender']), inplace=True)
df['Married'].fillna(stats.mode(df['Married']), inplace=True)
df['Dependents'].fillna(stats.mode(df['Dependents']), inplace=True)
df['Self_Employed'].fillna(stats.mode(df['Self_Employed']), inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(), inplace=True)
df['Credit_History'].fillna(stats.mode(df['Credit_History']), inplace=True)

# Reading the dataset in a dataframe using Pandas
allPredictors = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']
for i in allPredictors:
    df[i].fillna(stats.mode(df[i]), inplace=True)

var_mod = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])
print(df)

# Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
    # Fit the model:
    model.fit(data[predictors], data[outcome])

    # Make predictions on training set:
    predictions = model.predict(data[predictors])

    # Print accuracy
    accuracy = metrics.accuracy_score(predictions, data[outcome])
    print ("Training accuracy : %s % {0:.3%}".format(accuracy))

    # Perform k-fold cross-validation with 5 folds
    kf = KFold(n_splits=10)
    accuracy = []
    for train, test in kf.split(data):
        # Filter training data
        train_predictors = (data[predictors].iloc[train, :])

        # The target we're using to train the algorithm.
        train_target = data[outcome].iloc[train]

        # Training the algorithm using the predictors and target.
        model.fit(train_predictors, train_target)

        # Record accuracy from each cross-validation run
        accuracy.append(model.score(data[predictors].iloc[test, :], data[outcome].iloc[test]))

    print ("Cross-Validation Score : %s % {0:.3%}".format(np.mean(accuracy)))

    # Fit the model again so that it can be refered outside the function:
    model.fit(data[predictors], data[outcome])

    bestPred = pd.Series(model.feature_importances_, index=predictors).sort_values(ascending=False)

    return bestPred

# Apply Random Forest with improved predictors and parameters
print("\nRandom:")
myModel = RandomForestClassifier(n_estimators=100)
predictor_var = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']
classification_model(myModel, df, predictor_var, 'Loan_Status')

print("\nUsing best:")
bestModel = RandomForestClassifier(n_estimators=100, max_features='auto')
bestPredicts = ['Credit_History', 'ApplicantIncome', 'LoanAmount', 'CoapplicantIncome', 'Dependents']
classification_model(bestModel, df, bestPredicts, 'Loan_Status')
