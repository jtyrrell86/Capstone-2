import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import math



def mulitmodels(model, df):
    X = df.drop("meter_reading", axis=1)
    y = df["meter_reading"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100)
    model.fit(X_train.drop("row_id", axis=1), y_train)
    y_pred = model.predict(X_test.drop("row_id"))
    return model.score(X_test.drop("row_id"), y_test), precision_score(y_test, y_pred), recall_score(y_test, y_pred), y_pred, y_test

def mulitmodels_for_kaggle(model, df, X_test):
    X = df.drop("meter_reading", axis=1)
    y = df["meter_reading"]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100)
    model.fit(X.drop("row_id", axis=1), y)
    y_pred = model.predict(X_test.drop("row_id"))
    return y_pred

def rmsle(y_pred, y_test):
    res = 0
    for pred, actual in zip(y_pred, y_test):
        res += (math.log(pred + 1) - math.log(actual + 1))**2
    return math.sqrt(res / y_pred.shape()[0])

def target_normalization(df):
    df["meter_readings"] = np.log(["meter_readings"])
    return df

def feature_normalization():
    # This function will normalize features such as sqft and temp using (val - mean) /std
    pass

def some_plot():
    '''
    Plots:
    Be sure to add a visual of my data to the readme
    1) mean usage vs. time of day
    2) pie chart of the usage type
    3) usage of the 3-4 different meter types on one graph
    4) significance plots
    5) ?
    '''

if __name__ == "__main__":
    cleaned_df = pd.read_csv("../data/cleaned_df.csv")
    cleaned_df.drop("Unnamed: 0", axis=1, inplace=True)
    score, precisions_score, recall_score, y_pred, y_test = mulitmodels(DecisionTreeClassifier(), cleaned_df)
    # rmsle_score = rmsle(y_pred, y_test)
    # print(rmsle_score, score, precisions_score, recall_score)
    
    
