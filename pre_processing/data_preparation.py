repo_path = r"C:\Users\ssrikrishnan6\Metalearning_Survival_Analysis\metalearning_survival/"
# repo_path = "/home/amandal36/sreenath/metalearning_survival/"

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

  X = df[feature_list]
  y = df[["time","status"]]
  if title == "MESO":
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.20, random_state=42)
  else:
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, train_size=20, random_state=42)

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
  exclude_list = ["GBM", "LGG", "LUAD", "LUSC", "HNSC", "MESO"]
  meta_train = df.drop(df.index[df['cancer_type'].isin(exclude_list)])

  # Target Cancer types -  20 samples each of (GBB, LGG, LUAD, LUSC, HNSC) , All samples of MESO
  meta_test_GBM = df.drop(df.index[df['cancer_type'] != "GBM"])
  meta_test_LGG = df.drop(df.index[df['cancer_type'] != "LGG"])
  meta_test_LUAD = df.drop(df.index[df['cancer_type'] != "LUAD"])
  meta_test_LUSC = df.drop(df.index[df['cancer_type'] != "LUSC"])
  meta_test_HNSC = df.drop(df.index[df['cancer_type'] != "HNSC"])
  meta_test_MESO = df.drop(df.index[df['cancer_type'] != "MESO"])

  # split_train_test(meta_train, "protein_pancan_v1", "metatrain")
  split_train_test(meta_test_MESO, "MESO", "metatest")
  split_train_test(meta_test_GBM, "GBM", "metatest")
  split_train_test(meta_test_LGG, "LGG", "metatest")
  split_train_test(meta_test_LUAD, "LUAD", "metatest")
  split_train_test(meta_test_LUSC, "LUSC", "metatest")
  split_train_test(meta_test_HNSC, "HNSC", "metatest")

def list_of_features(df):
  feature_list = df.columns.tolist()
  feature_list.remove("time")
  feature_list.remove("status")
  feature_list.remove("cancer_type")
  return feature_list

def scale_removenan_fillnan(df):
  ##Normalizing the data using min-max scaling
  from sklearn.preprocessing import minmax_scale
  from sklearn.preprocessing import MinMaxScaler

  df.dropna(subset = ["time", "status"], inplace=True) #Removing na cells for time and stauts by Subsetting rows in pandas
  df = df.loc[df['time'] != 0]
  df.dropna()#(how='all') #df.dropna(thresh=300)

  ##Normalization using MinMaxScaler in range of (-1,1)
  feature_list = list_of_features(df)
  scaler = MinMaxScaler(feature_range=(-1,1))
  df[feature_list] = scaler.fit_transform(df[feature_list])

  #df[feature_list] = minmax_scale(df[feature_list])
  # print("Number of rows without missing cells", df.shape[0] - df.dropna().shape[0])
  df = df.fillna(df.mean())
  return df

  ###Option from previous work
  # from sklearn.preprocessing import StandardScaler
  # counts_data = protein_expression_tcga[feature_list].T
  # scaler = StandardScaler()
  # scaler.fit(counts_data)
  # scaled_data = scaler.transform(counts_data)
  # scaled_data.shape

def main():
  import pandas as pd
  protein_expression_tcga = pd.read_csv(repo_path + 'pre_processing/tcga_protein_df.csv', index_col=0)
  # microrna_expression_tcga = pd.read_csv(repo_path + 'pre_processing/tcga_microrna_df.csv', index_col=0)

  # print(protein_expression_tcga.shape)
  protein_expression_tcga = scale_removenan_fillnan(protein_expression_tcga)
  protein_expression_tcga.to_csv("protein_expression_tcga.csv")

  ###TODO: Index has duplicate values??? Cross-check the MicroRNA Dataframe
  # microrna_expression_tcga.shape
  # microrna_feature_list = list_of_features(microrna_expression_tcga)
  # microrna_expression_tcga = scale_remove_nan(microrna_expression_tcga, microrna_feature_list)
  # microrna_expression_tcga.shape

  create_metatrain_metatest_data(protein_expression_tcga)

main()
