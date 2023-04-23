import os
import pandas as pd

tasks_name_directory = r'.\tasks'

# read csv at path and convert to df
weight_csv_file_path = os.path.join(tasks_name_directory, "weight_log.csv")
weightcsv_df = pd.read_csv(weight_csv_file_path)
# Replace NaN values with None
weightcsv_df = weightcsv_df.where(pd.notnull(weightcsv_df), None)
weightcsv_df = weightcsv_df.fillna('')
# print(f'weight csv:\n{weightcsv_df.dtypes}\n{weightcsv_df}\n')
print(weightcsv_df.columns.tolist())
# Update type
weightcsv_df['Date'] = weightcsv_df['Date'].astype(str)
weightcsv_df['Weight  '] = weightcsv_df['Weight  '].astype(str)
# weightcsv_df['Date'] = pd.to_datetime(weightcsv_df['Date'], format='%d-%m-%Y').dt.strftime('%d-%m-%Y')
print(f'weight csv:\n{weightcsv_df.dtypes}\n{weightcsv_df}\n')