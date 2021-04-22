import pandas as pd
import datetime
import time


def merge_dataframes(df1, df2, df3):
    '''Merges the three input dataframes on their primary and foreign keys

    Parameters
    ----------
    df1 : Pandas dataframe
        First dataframe to be merged
    df2 : Pandas dataframe
        Second dataframe to be merged on the right to preserve all rows from
        this dataframe
    df3 : Pandas dataframe
        Third dataframe to be merged on the right to preserve all rows in the
        dataframe this is being merged with

    Returns
    -------
    Pandas dataframe
        Merged dataframe
    '''
    new_df = pd.merge(left=df1, right=df2, how="right")
    new_df = pd.merge(left=new_df, right=df3, how="left")
    return new_df


def drop_initial_unused_cols(df, lst_of_cols_to_drop_1):
    '''This drops a list a columns defined under if name == main that are
    determined to not be needed or useful in modeling

    Parameters
    ----------
    df : Pandas dataframe
    lst_of_cols_to_drop_1 : list
        Initial columns to drop

    Returns
    -------
    Pandas dataframe
        Input dataframe with columns dropped
    '''
    return df.drop(lst_of_cols_to_drop_1, axis=1)


def create_quarterly_dummies_for_year(df):
    '''First converts timestamp to datetime format and creates a date column.
    Next it creates four masks associated with quarters of the year and labels
    these q1, q2, q3, and q4. It then applies these masks to the values in the
    date column converting their values to either q1, q2, q3, or q4. Lastly,
    it using one-hot encoding to convert categorical date column into four
    numeric dummy columns and drops the date column

    Parameters
    ----------
    df : Pandas dataframe
        Dataframe needing dates from the datetime column converted to dummy
        columns

    Returns
    -------
    Pandas dataframe
        Dataframe with four numeric dummy columns for the quarters of a year
        added
    '''
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    q1 = (df["date"] >= pd.to_datetime("2016-01-01").date()) & \
         (df["date"] < pd.to_datetime("2016-04-01").date())
    q2 = (df["date"] >= pd.to_datetime("2016-04-01").date()) & \
         (df["date"] < pd.to_datetime("2016-07-01").date())
    q3 = (df["date"] >= pd.to_datetime("2016-07-01").date()) & \
         (df["date"] < pd.to_datetime("2016-10-01").date())
    q4 = (df["date"] >= pd.to_datetime("2016-10-01").date()) & \
         (df["date"] <= pd.to_datetime("2016-12-31").date())
    df["date"][q1] = "q1"
    df["date"][q2] = "q2"
    df["date"][q3] = "q3"
    df["date"][q4] = "q4"
    date_dummies = pd.get_dummies(df["date"])
    df = df.join(other=date_dummies)
    df.drop(["date"], axis=1, inplace=True)
    return df


def create_quarterly_dummies_for_a_day(df):
    '''First it uses the timestamp column to create a time column.
    Next it creates four masks associated with quarters of the day and labels
    these early morning, morning, afternoon, and evening. It then applies these
    masks to the values in the time column converting their values to either
    early morning, morning, afternoon, and evening. Lastly, it using one-hot
    encoding to convert categorical time column into four numeric dummy columns
    and drops the time column

    Parameters
    ----------
    df : Pandas dataframe
        Dataframe needing times from the datetime column converted to dummy
        columns

    Returns
    -------
    Pandas dataframe
        Dataframe with four numeric dummy columns for the quarters of a day
        added
    '''

    df["time"] = df["timestamp"].dt.time
    early_morning = (df["time"] >= pd.to_datetime("00:00:00").time()) & \
                    (df["time"] < pd.to_datetime("06:00:00").time())
    morning = (df["time"] >= pd.to_datetime("06:00:00").time()) & \
              (df["time"] < pd.to_datetime("12:00:00").time())
    afternoon = (df["time"] >= pd.to_datetime("12:00:00").time()) & \
                (df["time"] < pd.to_datetime("18:00:00").time())
    evening = (df["time"] >= pd.to_datetime("18:00:00").time()) & \
              (df["time"] <= pd.to_datetime("23:59:00").time())
    df["time"][early_morning] = "early_morning"
    df["time"][morning] = "morning"
    df["time"][afternoon] = "afternoon"
    df["time"][evening] = "evening"
    time_dummies = pd.get_dummies(df["time"])
    df = df.join(other=time_dummies)
    df.drop(["time"], axis=1, inplace=True)
    return df


def create_primary_usage_and_meter_dummies(df):
    '''Creates dummy columns for the primary usage and meter columns and then
    deletes those original two columns

    Parameters
    ----------
    df : Pandas dataframe
        Input dataframe needing columns converted to dummy columns

    Returns
    -------
    Pandas dataframe
        Dataframe with columns converted to dummies and original columns
        removed
    '''
    meter_dummies = pd.get_dummies(df["meter"])
    df = df.join(other=meter_dummies)
    df = df.rename(
        columns={0: "electricity", 1: "chilledwater", 2: "steam", 3: "hotwater"})
    df.drop(["meter"], axis=1, inplace=True)

    usage_dummies = pd.get_dummies(df["primary_use"])
    df = df.join(other=usage_dummies)
    df.drop(["primary_use", "Religious worship"], axis=1, inplace=True)
    return df


def convert_site_one_meter_readings(df, to_kWh=True):
    '''Meter readings for building in site 0 were provided in different units
    then the rest of the sites. This function allows you to switch between Kbtu,
    which is how they were reported and how you'll need to submit them to Kaggle,
    and kWh, which you'll need to use for modeling

    Parameters
    ----------
    df : pandas dataframe
        Datframe that needs it's meter reading units for site 0 converted.
    to_kWh : bool, optional
        By default True. If left as such will convert from Kbtu to kWh. Else it
        will convert from kWh to Kbtu

    Returns
    -------
    Pandas dataframe
        Dataframe with values in the meter reading column for site 0 converted
        to either kWh or Kbtu
    '''
    if to_kWh:
        df["meter_reading"].loc[df["site_id"] == 0] = df.loc[df["site_id"] == 0, \
        "meter_reading"].apply(lambda x: x*0.2931)
    else:
        df["meter_reading"].loc[df["site_id"] == 0] = df.loc[df["site_id"] == 0, \
        "meter_reading"].apply(lambda x: x*3.4118)
    return df


def create_temp_mean_df(weather_df):
    '''This function uses the original weather dataframe to build a lookup table
    with the mean temperature for each day for a year and 16 different sites.
    The lookup table has a multi-index which stores both the site id and date.

    Parameters
    ----------
    weather_df : Pandas dataframe
        Original weather dataframe

    Returns
    -------
    Pandas dataframe
        Dataframe with mean temperatures
    '''
    weather_df_copy = weather_df.copy()
    weather_df_copy["timestamp"] = pd.to_datetime(weather_train_df["timestamp"])
    weather_df_copy["date"] = weather_df_copy["timestamp"].dt.date
    weather_df_copy.drop(["cloud_coverage", "dew_temperature", \
    "precip_depth_1_hr", "sea_level_pressure", "wind_direction", "wind_speed", \
    "timestamp"], axis=1, inplace=True)
    temp_mean_df = weather_df_copy.groupby(by=["site_id", "date"]).mean()
    return temp_mean_df


def impute_temp_nans(df, temp_mean_df):
    ''' This function iterates through the input dataframes air temperature
    column looking for nans. When it finds them it uses the date and site id to
    identify the correct mean temperature in the above generated table to
    replace it with. This may take upwards of 6.5 hours to run

    Parameters
    ----------
    df : Pandas dataframe
        Dataframe needing air temperature nan's imputed
    temp_mean_df : Pandas dataframe
        Dataframe with mean daily temperatures for 16 different sites

    Returns
    -------
    Pandas dataframe
        Dataframe with air temperature nan's imputed
    '''
    start = time.time()
    for idx, temp in enumerate(df["air_temperature"]):
        if pd.isna(temp):
            year = df.iloc[idx, 3].year
            month = df.iloc[idx, 3].month
            day = df.iloc[idx, 3].day
            df.iloc[idx, 5] = temp_mean_df.loc[(df.iloc[idx, 0], datetime.date \
                                (year, month, day))]["air_temperature"]
        if idx % 100000 == 0:
            checkpoint = time.time()
            print(f'{idx} 100,000 time:{checkpoint-start}')
    return df


def create_ref_col_and_drop_remaining_unused(df, list_of_cols_to_drop_2):
    '''This function drops a list of any remaining columns not needed for
    modeling. I also adds a reference column used for tracking

    Parameters
    ----------
    df : Pandas dataframe
        Dataframe needing to be prepared for modeling
    lst_of_cols_to_drop_2 : list
        Final columns to drop

    Returns
    -------
    Pandas dataframe
        Dataframe ready for modeling
    '''

    df = df.drop(list_of_cols_to_drop_2, axis=1)
    if "row_id" not in df.columns:
        df["row_id"] = df.index

    return df


def meter_type_subset(df, meter_type):
    return df[df[meter_type] == 1]


if __name__ == "__main__":
    metadata_df = pd.read_csv(
        "../data/ashrae-energy-prediction/building_metadata.csv")
    weather_train_df = pd.read_csv(
        "../data/ashrae-energy-prediction/weather_train.csv")
    train_df = pd.read_csv("../data/ashrae-energy-prediction/train.csv")

    weather_test_df = pd.read_csv(
        "../data/ashrae-energy-prediction/weather_test.csv")
    test_df = pd.read_csv("../data/ashrae-energy-prediction/test.csv")

    # Cleaning the training data
    '''combined_df = merge_dataframes(metadata_df, train_df, weather_train_df)
    lst_of_cols_to_drop_1 = ["cloud_coverage", "dew_temperature", \
        "precip_depth_1_hr" , "sea_level_pressure", "wind_direction", "wind_speed",\
        "year_built", "floor_count"]
    combined_df = drop_initial_unused_cols(combined_df, lst_of_cols_to_drop_1)
    combined_df = create_quarterly_dummies_for_year(combined_df)
    combined_df = create_quarterly_dummies_for_a_day(combined_df)
    combined_df = create_primary_usage_and_meter_dummies(combined_df)
    combined_df = convert_site_one_meter_readings(combined_df, to_kWh=True)
    temp_mean_df = create_temp_mean_df(weather_train_df)
    # combined_df = impute_temp_nans(combined_df, temp_mean_df)
    list_of_cols_to_drop_2 = ["timestamp", "site_id", "building_id"]
    cleaned_df = create_ref_col_and_drop_remaining_unused(combined_df, \
        list_of_cols_to_drop_2)
    # cleaned_df.to_csv("../data/cleaned_df.csv")'''

    # Splitting the data into meter type for individual modeling
    '''cleaned_df = pd.read_csv("../data/cleaned_df.csv") # Delete this for main branch
    electricity_subset = meter_type_subset(cleaned_df, "electricity")
    electricity_subset.to_csv("../data/electricity_subset.csv")

    chilledwater_subset = meter_type_subset(cleaned_df, "chilledwater")
    chilledwater_subset.to_csv("../data/chilledwater_subset.csv")

    steam_subset = meter_type_subset(cleaned_df, "steam")
    steam_subset.to_csv("../data/steam_subset.csv")

    hotwater_subset = meter_type_subset(cleaned_df, "hotwater")
    hotwater_subset.to_csv("../data/hotwater_subset.csv")'''

    # Cleaning the test data
    combined_test_df = merge_dataframes(metadata_df, test_df, weather_test_df)
    lst_of_cols_to_drop_1 = ["cloud_coverage", "dew_temperature", \
        "precip_depth_1_hr", "sea_level_pressure", "wind_direction", "wind_speed",\
        "year_built", "floor_count"]
    combined_test_df = drop_initial_unused_cols(combined_test_df, \
        lst_of_cols_to_drop_1)
    combined_test_df = create_quarterly_dummies_for_year(combined_test_df)
    combined_test_df = create_quarterly_dummies_for_a_day(combined_test_df)
    combined_test_df = create_primary_usage_and_meter_dummies(combined_test_df)
    temp_mean_test_df = create_temp_mean_df(weather_test_df)
    combined_test_df = impute_temp_nans(combined_test_df, temp_mean_test_df)
    list_of_cols_to_drop_2 = ["timestamp", "site_id", "building_id"]
    cleaned_test_df = create_ref_col_and_drop_remaining_unused(combined_test_df, \
        list_of_cols_to_drop_2)
    cleaned_test_df.to_csv("../data/cleaned_test_df.csv")

    # Splitting the test data into meter type for individual modeling
    electricity_test_subset = meter_type_subset(cleaned_test_df, "electricity")
    electricity_test_subset.to_csv("../data/electricity_test_subset.csv")

    chilledwater_test_subset = meter_type_subset(cleaned_test_df, "chilledwater")
    chilledwater_test_subset.to_csv("../data/chilledwater_test_subset.csv")

    steam_test_subset = meter_type_subset(cleaned_test_df, "steam")
    steam_test_subset.to_csv("../data/steam_test_subset.csv")

    hotwater_test_subset = meter_type_subset(cleaned_test_df, "hotwater")
    hotwater_test_subset.to_csv("../data/hotwater_test_subset.csv")

    ############## CODE TESTING- DELETE ONCE PUSHED TO MAIN BRANCH#########