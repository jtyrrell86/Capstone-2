import pandas as pd
import numpy as np

df = pd.read_csv("Arapahoe_County_Weather_Data_March_1.csv", low_memory=False, dtype={"DailyHeatingDegreeDays": float})

# print(df["REPORT_TYPE"].unique())

df1 = df[df["REPORT_TYPE"] == "SOD  "]
# df1[""]
pd.set_option('display.max_columns', 124)
HDD = df1["DailyHeatingDegreeDays"].sum()
CDD = df1["DailyCoolingDegreeDays"].sum()
print(df1["DailyHeatingDegreeDays"].head(30))