## Creating a dataframe with all TCGA OMICS (Protein/MicroRNA) Expression Data, Survival Time and Survival Status ##
trgated_path = "TRGAted_csv/"
import os
import pandas as pd

repo_path = r"C:\Users\ssrikrishnan6\Metalearning_Survival_Analysis\metalearning_survival/"
# repo_path = "/Users/sreenath/metalearning_survival/"
def write_datasets(final_stage, path, X_train, X_holdout, ytime_train, ystatus_train, ytime_holdout, ystatus_holdout):
  X_train.to_csv(path+"_feature_train.csv",index=False)
  ytime_train.to_csv(path+"_ytime_train.csv",index=False)
  ystatus_train.to_csv(path+"_ystatus_train.csv",index=False)

  X_holdout.to_csv(path+"_feature_"+final_stage+".csv",index=False) ##For the metatrain stage the holdout is what is used for validation
  ytime_holdout.to_csv(path+"_ytime_"+final_stage+".csv",index=False)
  ystatus_holdout.to_csv(path+"_ystatus_"+final_stage+".csv",index=False)

def split_train_test(df,title,metastage):
  ###Splitting the Data into meta-traina and test and further into 80:20
  from sklearn.model_selection import train_test_split

  df.drop('cancer_type', axis=1, inplace=True)
  feature_list = df.columns.tolist()
  feature_list.remove("time")
  feature_list.remove("status")
  feature_list.remove("patient")

  X = df[feature_list]
  y = df[["time","status"]]

  X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.20, random_state=42)

  ytime_train = y_train["time"]
  ystatus_train = y_train["status"]
  ytime_holdout = y_holdout["time"]
  ystatus_holdout = y_holdout["status"]

  if metastage == "metatrain":
    path = repo_path + "sample_data/pretrainPanCan/" + title
    final_stage = "val"
  else:
    path = repo_path + "sample_data/finetuneTarget/" + title
    final_stage = "holdout"

  write_datasets(final_stage, path, X_train, X_holdout, ytime_train, ystatus_train, ytime_holdout, ystatus_holdout)

def create_metatrain_metatest_data(df):
  # Test Cancer - All except for GBM, LGG, LUAD, LUSC, HNSC, MESO
  df = df[df.cancer_type != "GBM"]
  df = df[df.cancer_type != "LGG"]
  df = df[df.cancer_type != "LUAD"]
  df = df[df.cancer_type != "LUSC"]
  df = df[df.cancer_type != "HNSC"]
  meta_train = df[df.cancer_type != "MESO"]

  # exclude_list = ["GBM", "LGG", "LUAD", "LUSC", "HNSC", "MESO"]
  # meta_train = df.drop(df.index[df['cancer_type'].isin(exclude_list)])
  # print("META TRAIN SHAPE ------", meta_train.shape)

  # Target Cancer types -  20 samples each of (GBB, LGG, LUAD, LUSC, HNSC) , All samples of MESO
  # meta_test_GBM = df.drop(df.index[df['cancer_type'] != "GBM"]).sample(n=30)
  # meta_test_LGG = df.drop(df.index[df['cancer_type'] != "LGG"]).sample(n=30)
  # meta_test_LUAD = df.drop(df.index[df['cancer_type'] != "LUAD"]).sample(n=30)
  # meta_test_LUSC = df.drop(df.index[df['cancer_type'] != "LUSC"]).sample(n=30)
  # meta_test_HNSC = df.drop(df.index[df['cancer_type'] != "HNSC"]).sample(n=30)
  # meta_test_MESO = df.drop(df.index[df['cancer_type'] != "MESO"])

  split_train_test(meta_train, "protein_pancan_trgated", "metatrain")
  # split_train_test(meta_test_MESO, "MESO_trgated", "metatest")
  # split_train_test(meta_test_GBM, "GBM_trgated", "metatest")
  # split_train_test(meta_test_LGG, "LGG_trgated", "metatest")
  # split_train_test(meta_test_LUAD, "LUAD_trgated", "metatest")
  # split_train_test(meta_test_LUSC, "LUSC_trgated", "metatest")
  # split_train_test(meta_test_HNSC, "HNSC_trgated", "metatest")

def list_of_features(df):
  feature_list = df.columns.tolist()
  feature_list.remove("time")
  feature_list.remove("status")
  feature_list.remove("cancer_type")
  feature_list.remove('patient')
  return feature_list

def scale_removenan_fillnan(df):
  ##Normalizing the data using min-max scaling
  from sklearn.preprocessing import MinMaxScaler

  df.dropna(subset = ["time", "status"], inplace=True) #Removing na cells for time and stauts by Subsetting rows in pandas
  df = df.loc[df['time'] != 0]
  df.dropna()#(how='all') #df.dropna(thresh=300)

  ##Normalization using MinMaxScaler in range of (-1,1)
  feature_list = list_of_features(df)
  scaler = MinMaxScaler(feature_range=(-1,1))
  df[feature_list] = scaler.fit_transform(df[feature_list])

  df = df.fillna(df.mean())
  return df

def create_combined_rppa_df():
  merge_df = []

  for file in os.listdir("TRGAted_csv/"):
    try:
      df = pd.read_csv("TRGAted_csv/" + file)
      df.rename(columns={"OS": "status", "OS.time": "time"}, inplace=True)
      df["cancer_type"] = file.rstrip(".csv")
      columns = df.columns.values.tolist()
      to_remove = ['subtype', 'age', 'gender', 'race', 'ajcc_pathologic_tumor_stage', 'histological_type',
                   'histological_grade', 'treatment_outcome_first_course', 'DSS', 'DSS.time', 'DFI', 'DFI.time', 'PFI',
                   'PFI.time']
      for element in to_remove:
        columns.remove(element)
      df = df[columns]
      merge_df.append(df)
      print("Success for ", file)

    except:
      print("Failed for ", file)

  master_df = pd.DataFrame()  # The braces are to prevent error: The reason behind the error is the first argument of a method is the class instance.
  master_df = master_df.append(merge_df, sort=False)
  master_df.to_csv("pre_processing/tcga_protein_trgated_df.csv")

def main():
  create_combined_rppa_df()
  protein_expression_tcga = pd.read_csv(repo_path + 'pre_processing/tcga_protein_trgated_df.csv', index_col=0)
  protein_expression_tcga = scale_removenan_fillnan(protein_expression_tcga)

  # print(protein_expression_tcga.groupby(['cancer_type']).size())
  # protein_expression_tcga.to_csv("protein_expression_tcga_trgated.csv")
  create_metatrain_metatest_data(protein_expression_tcga)

main()
