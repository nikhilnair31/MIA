import os
import gspread
import pandas as pd

gc = gspread.service_account("gs_credentials.json")

dumpfile = gc.open("Routine 2023 Dump")
dumpfile_weightsheet = dumpfile.worksheet("Weight")
print(f'dumpfile_weightsheet: {dumpfile_weightsheet.get_all_values()}\n')

# read csv at path and convert to df
weightcsv_df = pd.read_csv(os.path.join(r'.\tasks', "weight_log.csv"))
# Replace NaN values with None
weightcsv_df = weightcsv_df.where(pd.notnull(weightcsv_df), None)
weightcsv_df = weightcsv_df.fillna('')
print(f'weight csv: {weightcsv_df}\n')

# export df to a dumpfile_weightsheet
dumpfile_weightsheet.update([weightcsv_df.columns.values.tolist()] + weightcsv_df.values.tolist())