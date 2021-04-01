import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import math

def mulitmodels(model, df):
    '''
    Allows the user to pass in any model type, as well as the data, and recieve an r^2 score, y_predictions,
     y_test, and feature importances. The later three can be used in future calculations or plotting.
    '''
    X = df.drop("meter_reading", axis=1)
    y = df["meter_reading"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model.score(X_test, y_test), y_pred, y_test, model.feature_importances_

def mulitmodels_for_kaggle(model, df, X_test):
    '''
    Similiar to the function above, but this will all you pass in X_test as well. This function uses the full
    dataset to fit the model and then uses the x_test to make predictions. The returned predictions can then be
    submitted to kaggle for scoring. It still needs to be formatted correctly though.
    '''
    X = df.drop("meter_reading", axis=1)
    y = df["meter_reading"]
    model.fit(X.drop("row_id", axis=1), y)
    y_pred = model.predict(X_test.drop("row_id", axis=1))
    return y_pred

def rmsle(y_pred, y_test):
    '''
    Calculated root mean squared log error (rmsle)
    '''
    res = 0
    for pred, actual in zip(y_pred, y_test):
        res += (np.log(pred + 1) - np.log(actual + 1))**2
    return math.sqrt(res / len(y_pred))

def rmse(y_pred, y_test):
    '''
    Calsulates root mean squared error for target data the has already been log transformed.
    '''
    res = 0
    for pred, actual in zip(y_pred, y_test):
        res += (pred - actual)**2
    return math.sqrt(res / len(y_pred))

def target_transformation(df):
    '''
    Log transforms the target data.
    '''
    df["meter_reading"].loc[df["meter_reading"] >= 0] = df.loc[df["meter_reading"] >= 0, "meter_reading"].apply(lambda x: np.log(x+1))
    return df

def feature_normalization():
    # This function will normalize features such as sqft and temp using (val - mean) /std
    pass

def feature_significance_plot(feature_names, feature_importances):
    '''
    Will take in the top to features and their significance and plot them.
    '''
    plt.title("Feature Importances for Top 5")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.bar(feature_names, feature_importances, color="r")
    plt.tight_layout()
    plt.savefig("../images/feature_importance_plot")

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
    # cleaned_df = pd.read_csv("../data/cleaned_df.csv")
    # cleaned_df.drop("Unnamed: 0", axis=1, inplace=True)
    hotwater_df = pd.read_csv("../data/hotwater_df.csv")
    # hotwater_df_0 = hotwater_df.drop(["Unnamed: 0", "row_id"], axis=1)
    hotwater_df_0 = hotwater_df.drop(["Unnamed: 0", "row_id", "electricity","chilledwater", "steam", "hotwater", "Manufacturing/industrial","Other",
       "Parking", "Retail", "Services", "Utility", "Warehouse/storage"], axis=1)
    hotwater_df_1 = hotwater_df.drop(["Unnamed: 0", "row_id", "electricity","chilledwater", "steam", "hotwater", "Manufacturing/industrial","Other",
       "Parking", "Retail", "Services", "Utility", "Warehouse/storage"], axis=1)
    hotwater_df_1 = target_transformation(hotwater_df_1)

    # rf_0 = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    # score_0, y_pred_0, y_test_0, feat_importances_0 = mulitmodels(rf_0, hotwater_df_0)
    # rmsle_score_0 = rmsle(y_pred_0, y_test_0)
    # sorted_feature_indices_0 = np.argsort(feat_importances_0)[::-1]
    # feature_names_0 = hotwater_df_0.drop("meter_reading", axis=1).columns
    # top_5_features_0 = feature_names_0[sorted_feature_indices_0][:]

    rf_1 = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    score_1, y_pred_1, y_test_1, feat_importances_1 = mulitmodels(rf_1, hotwater_df_1)
    rmse_score_1 = rmse(y_pred_1, y_test_1)
    sorted_feature_indices_1 = np.argsort(feat_importances_1)[::-1]
    feature_names_1 = hotwater_df_1.drop("meter_reading", axis=1).columns
    top_5_features_1 = feature_names_1[sorted_feature_indices_1][:5]
    top_5_features_1_significances = feat_importances_1[sorted_feature_indices_1][:5]


    # print(rmsle_score_0, score_0, top_5_features_0)
    # print(rmse_score_1, score_1, top_5_features_1)
    print(feature_significance_plot(top_5_features_1, top_5_features_1_significances))

    
    
    
