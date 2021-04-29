import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import math


def primary_use_bar_graph(metadata, labels):
    '''Takes in the metadata dataframe and a list of labels for the top six
    most common primary building uses. The first five our the most common
    whereas other is made up of the remaining primary use type

    Parameters
    ----------
    metadata : Pandas dataframe
        Dataframe that includes information about 1449 buildings including
        their primary usage
    labels : List of primary use types as described above

    Returns
    -------
    primary_use_bar_graph : .png file
    '''

    primary_use = metadata.groupby(by="primary_use").count()["building_id"]
    primary_use = primary_use.sort_values(ascending=False)
    other = pd.Series([primary_use[5:].sum()], index=(["Other"]))
    primary_use = primary_use[:5].append(other)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    ax.set_title("Primary Uses", fontsize=18, color="black")
    ax.set_ylabel("# of Buildings", fontsize=14, color="black")
    ax.tick_params(axis='y', colors='black')
    ax.tick_params(axis='x', colors='black')
    ax.bar(labels, primary_use.values)
    fig.tight_layout(pad=1)
    plt.savefig(f"../images/primary_use_bar_graph.png")


def percentage_meter_readings_by_energy__type_plot(train_df):
    '''Takes in the dataframe containing all meter readings, calculates the
    percentage of meter readings by meter type, and plots that on a stacked
    bar graph

    Parameters
    ----------
    train_df : Dataframe
        Dataframe containing all meter readings

    Returns
    -------
    percentage_meter_readings_by_energy__type_plot : .png file
    '''

    percent_electricity = round((train_df[train_df["meter"] == 0]["meter"].
        count() / train_df["meter"].count() * 100), 2)
    percent_chilled_water = round((train_df[train_df["meter"] == 1]["meter"].
        count() / train_df["meter"].count() * 100), 2)
    percent_steam = round((train_df[train_df["meter"] == 2]["meter"].
        count() / train_df["meter"].count() * 100), 2)
    percent_hot_water = round((train_df[train_df["meter"] == 3]["meter"].
        count() / train_df["meter"].count() * 100), 2)

    series = pd.Series([percent_electricity, percent_chilled_water,
        percent_steam, percent_hot_water], index=["Electricity",
        "Chilled Water", "Steam", "Hot Water"])
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    pd.DataFrame(series).T.plot.bar(stacked=True, ax=ax)
    plt.style.use('ggplot')
    ax.set_title(
        "Percentage Meter Readings by\n Metered Energy Type (Train Data)",
        fontsize=18, color="black")
    ax.set_ylabel("Percent", fontsize=14, color="black")
    ax.xaxis.label.set_color('black')
    ax.tick_params(axis='y', colors='black')
    plt.xticks(visible=False)
    plt.savefig("../images/meter_type_distribution.png", bbox_inches='tight')


def drop_unimportant_columns(df, drop_list):
    '''Takes in a dataframe and a drop list and drops the columns on the list

    Parameters
    ----------
    df : Pandas dataframe
        Dataframe needing columns removed
    drop_list : list
        Customizable list of columns to drop. Reasons for dropping include
        columns created unnecessarily, like the Unnamed: 0 columns, created
        by the cleaning script, columns that cannot be used in the model, and
        columns deemed unimport by during modeling

    Returns
    -------
    df: Pandas dataframe
        Datframe with selected columns removed
    '''

    df.drop(drop_list, axis=1, inplace=True)
    return df


def mulitmodels(model, df):
    ''' Allows the user to pass in an instance of any model type, as well as
    the data. It then does a train, test, split and fits the model on the train
    data. Lastly, predicts on the test data and returns an r^2 score,
    y_predictions, y_test, and feature importances. The later three can be used
    in future calculations or plotting.

    Parameters
    ----------
    model : Model instance
        Already instantiated instance of any model
    df : Pandas dataframe
        Dataframe that will be used in the model

    Returns
    -------
    model.score(X_test, y_test): Float
        Coefficient of determination or r^2 score
    y_pred: Numpy array
        Model predictions of the target value, meter_reading
    y_test: Numpy array
        Actual target values
    feature_importances_: Numpy array
        The impurity-based feature importances
    '''
    X = df.drop("meter_reading", axis=1)
    y = df["meter_reading"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model.score(X_test, y_test), y_pred, y_test, \
        model.feature_importances_


def mulitmodels_for_kaggle(model, df, X_test):
    ''' Allows the user to pass in an instance of any model type, the train
    dataframe, and the test dataframe. It then trains the model using the
    full train dataframe and predicts on the test dataframe. Lastly, it returns
    y_predictions that can be uploaded to kaggle.

    Parameters
    ----------
    model : Model instance
        Already instantiated instance of any model
    df : Pandas dataframe
        Train dataframe that will be used in the model
    X_test : Pandas dataframe
        Test dataframe that does not have the target included

    Returns
    -------
    y_pred: Numpy array
        Model predictions of the target value, meter_reading
    '''
    X = df.drop("meter_reading", axis=1)
    y = df["meter_reading"]
    model.fit(X.drop("row_id", axis=1), y)
    y_pred = model.predict(X_test.drop("row_id", axis=1))
    return y_pred


def rmsle(y_pred, y_test):
    '''Calculates the root mean squared log error, or rmsle, of the predicted
    target values (y_pred) and the actual target values (y_test)

    Parameters
    ----------
    y_pred: Numpy array
        Model predictions of the target value, meter_reading
    y_test: Numpy array
        Actual target values

    Returns
    -------
    calculated rmsle: Float
        Calculated rmsle
    '''

    res = 0
    for pred, actual in zip(y_pred, y_test):
        res += (np.log(pred + 1) - np.log(actual + 1))**2
    return math.sqrt(res / len(y_pred))


def rmse(y_pred, y_test):
    '''Calculates the root mean squared error, or rmse, of the predicted
    target values (y_pred) and the actual target values (y_test). This is
    specifically for instances when the target values have already been
    log transformed

    Parameters
    ----------
    y_pred: Numpy array
        Model predictions of the target value, meter_reading.
    y_test: Numpy array
        Actual target values

    Returns
    -------
    calculated rmse: Float
        Calculated rmse
    '''

    res = 0
    for pred, actual in zip(y_pred, y_test):
        res += (pred - actual)**2
    return math.sqrt(res / len(y_pred))


def target_log_transformation(df):
    '''Takes in a dataframe that includes the target column, meter reading,
    and log transforms the values if the default log=True is left untouched.
    If it's changed to false it will reverse the log transformation.

    Parameters
    ----------
    df : Pandas dataframe
        Dataframe that includes the target column, meter reading

    Returns
    -------
    Pandas dataframe
        Dataframe with the target column, meter reading, log transformed
    '''
    df["meter_reading"].loc[df["meter_reading"] >= 0] = df.loc[df[
        "meter_reading"] >= 0, "meter_reading"].apply(lambda x: np.log(x+1))
    return df


def reverse_log_transformation(sr):
    sr[sr >= 0] = sr[sr >= 0].apply(lambda x: np.expm1(x))
    return sr


def feature_normalization(df):
    '''This function will take in a dataframe and normalize the values in the
    sqft and air temperature columns

    Parameters
    ----------
    df : Pandas dataframe
        Dataframe that includes sqft and air temperature

    Returns
    -------
    Pandas dataframe
        Dataframe with the sqft and air temperature values normalized
    '''

    pass


def feature_importance_bar_graph(meter_type, feature_importances, df):
    '''Will take in the meter type being modeled, the feature importances
    and the dataframe used used for modeling. It will then extract the top
    five most important features, and their names, and plot them.

    Parameters
    ----------
    meter_type : String
        The name of the meter type being modeled
    feature_importances : Numpy array
        Importance values for all features in the model
    df : Pandas dataframe
        The dataframe being modeled

    Returns
    -------
    feature_importance_bar_graph : .png file
    '''

    sorted_feature_indices = np.argsort(feature_importances)[::-1]
    feature_names = pd.Series(df.drop("meter_reading", axis=1).columns)
    top_5_feature_names = feature_names[sorted_feature_indices][:5]
    for feature in top_5_feature_names:
        top_5_feature_names.replace(feature, feature.replace("_", " "),
            inplace=True)
    top_5_feature_importances = feature_importances[sorted_feature_indices][:5]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    ax.set_title(f"{meter_type} Feature Importances", fontsize=18,
        color="black")
    ax.set_xlabel("Features", fontsize=14, color="black")
    ax.set_ylabel("Importance", fontsize=14, color="black")
    ax.tick_params(axis='y', colors='black')
    ax.tick_params(axis='x', colors='black')
    ax.bar(top_5_feature_names, top_5_feature_importances)
    fig.tight_layout(pad=1)
    plt.savefig(f"../images/{meter_type}_feature_importance_bar_graph.png")


if __name__ == "__main__":
    # Full cleaned training set
    cleaned_df = pd.read_csv("../data/cleaned_df.csv")
    cleaned_df.drop("Unnamed: 0", axis=1, inplace=True)

    # Data for EDA plotting
    metadata_df = pd.read_csv("../data/\
        ashrae-energy-prediction/building_metadata.csv")
    plt.style.use("ggplot")
    labels = ["Education", "Office", "Entertainment/ \n public assembly",
        "Public services", "Lodging/ \n residential", "Other"]
    primary_use_bar_graph(metadata_df, labels)

    # Cleaned training data for each meter type
    hotwater_subset = pd.read_csv("../data/hotwater_subset.csv",
        index_col="Unnamed: 0")
    electricity_subset = pd.read_csv("../data/electricity_subset.csv",
        index_col="Unnamed: 0")
    chilledwater_subset = pd.read_csv("../data/chilledwater_subset.csv",
        index_col="Unnamed: 0")
    steam_subset = pd.read_csv("../data/steam_subset.csv",
        index_col="Unnamed: 0")

    # Water meter type data to model
    water_combined = pd.concat([hotwater_subset, chilledwater_subset,
        steam_subset])
    water_combined_drop_list = ["Unnamed: 0.1", "row_id"]
    water_combined = drop_unimportant_columns(water_combined,
        water_combined_drop_list)
    water_combined.rename(columns={"q1": "1st_Quarter", "q2": "2nd_Quarter",
        "q3": "3rd_Quarter", "q4": "4th_Quarter"}, inplace=True)
    water_combined = target_log_transformation(water_combined)

    # Modeling water meter type data
    water_combined_rf = RandomForestRegressor(n_estimators=100,
        n_jobs=-1)
    water_combined_score, water_combined_y_pred, water_combined_y_test, \
        water_combined_feature_importances = mulitmodels(water_combined_rf,
        water_combined)

    print(feature_importance_bar_graph("Water",
        water_combined_feature_importances, water_combined))

    # electricity data to model
    electricity_drop_list = ["Unnamed: 0.1", "row_id"]
    electricity_df = drop_unimportant_columns(electricity_subset,
        electricity_drop_list)
    electricity_df.rename(columns={"q1": "1st_Quarter", "q2": "2nd_Quarter",
        "q3": "3rd_Quarter", "q4": "4th_Quarter"}, inplace=True)
    electricity_df = target_log_transformation(electricity_df)

    # Modeling electricity data
    electricity_rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    electricity_score, electricity_y_pred, electricity_y_test, \
        electricity_feature_importances \
        = mulitmodels(electricity_rf, electricity_df)

    print(feature_importance_bar_graph("Electricity",
        electricity_feature_importances, electricity_df))

    # Combining subset predictions
    combined_y_pred = pd.Series(np.concatenate((water_combined_y_pred,
        electricity_y_pred), axis=0))
    combined_y_test = pd.concat([water_combined_y_test,
        electricity_y_test], axis=0)
    combined_y_pred = reverse_log_transformation(combined_y_pred)
    combined_y_test = reverse_log_transformation(combined_y_test)
    combined_rmsle = rmsle(combined_y_pred, combined_y_test)
    print(f"Combined RMSLE: {round(combined_rmsle, 3)}")
