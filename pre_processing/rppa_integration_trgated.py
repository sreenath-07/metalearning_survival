## Creating a dataframe with all TCGA OMICS (Protein/MicroRNA) Expression Data, Survival Time and Survival Status ##
trgated_path = "TRGAted_csv/"
import os
import pandas as pd

merge_df = []

def list_of_features(df):
  feature_list = df.columns.tolist()
  feature_list.remove("time")
  feature_list.remove("status")
  feature_list.remove("cancer_type")
  return feature_list

for file in os.listdir("TRGAted_csv/"):
  try:
    df = pd.read_csv("TRGAted_csv/"+file)
    df.rename(columns={"OS": "status", "OS.time": "time"}, inplace=True)
    df["cancer_type"] = file.rstrip(".csv")
    columns = df.columns.values.tolist()
    to_remove = ['subtype', 'age', 'gender', 'race', 'ajcc_pathologic_tumor_stage', 'histological_type', 'histological_grade', 'treatment_outcome_first_course', 'DSS', 'DSS.time', 'DFI', 'DFI.time', 'PFI', 'PFI.time']
    for element in to_remove:
      columns.remove(element)
    df = df[columns]
    merge_df.append(df)
    print("Success for ", file)

  except:
    print("Failed for ", file)

master_df = pd.DataFrame()  # The braces are to prevent error: The reason behind the error is the first argument of a method is the class instance.
master_df = master_df.append(merge_df, sort=False)

from sklearn.preprocessing import MinMaxScaler
feature_list = list_of_features(master_df)
scaler = MinMaxScaler(feature_range=(-1,1))
master_df[feature_list] = scaler.fit_transform(master_df[feature_list])

master_df.to_csv("pre_processing/tcga_protein_trgated_df.csv")

