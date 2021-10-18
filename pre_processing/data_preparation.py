repo_path = r"C:\Users\ssrikrishnan6\Metalearning_Survival_Analysis\metalearning_survival/"


import pandas as pd
protein_expression_tcga = pd.read_csv(repo_path+'pre_processing/tcga_protein_df.csv', index_col=0)
microrna_expression_tcga =  pd.read_csv(repo_path+'pre_processing/tcga_microrna_df.csv',index_col=0)


def list_of_features(df):
  feature_list = df.columns.tolist()
  feature_list.remove("time")
  feature_list.remove("status")
  feature_list.remove("cancer_type")
  return feature_list

def scale_removenan_fillnan(df):
  ##Normalizing the data using min-max scaling
  from sklearn.preprocessing import minmax_scale

  feature_list = list_of_features(df)
  df[feature_list] = minmax_scale(df[feature_list])
  df.dropna(subset = ["time", "status"], inplace=True) #Removing na cells for time and stauts by Subsetting rows in pandas
  df = protein_expression_tcga.loc[df['time'] != 0]
  df.dropna(how='all')
  # df.dropna(thresh=300)

  df = df.fillna(df.mean())
  return df

print(protein_expression_tcga.shape)
protein_expression_tcga = scale_removenan_fillnan(protein_expression_tcga)
print(protein_expression_tcga.shape)



###TODO: Index has duplicate values??? Cross-check the MicroRNA Dataframe

# microrna_expression_tcga.shape
# microrna_feature_list = list_of_features(microrna_expression_tcga)
# microrna_expression_tcga = scale_remove_nan(microrna_expression_tcga, microrna_feature_list)
# microrna_expression_tcga.shape

###Option from previous work
# from sklearn.preprocessing import StandardScaler
# counts_data = protein_expression_tcga[feature_list].T
# scaler = StandardScaler()
# scaler.fit(counts_data)
# scaled_data = scaler.transform(counts_data)
# scaled_data.shape

###Splitting the Data into meta-traina and test and further into 80:20
from sklearn.model_selection import train_test_split

def split_train_test(df,title,metastage):
  df.drop('cancer_type', axis=1, inplace=True)
  feature_list = df.columns.tolist()
  feature_list.remove("time")
  feature_list.remove("status")

  X = df[feature_list]
  y = df[["time","status"]]

  X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.20, random_state=42)

  ytime_train = y_train["time"]
  ystatus_train = y_train["status"]
  ytime_holdout = y_holdout["time"]
  ystatus_holdout = y_holdout["status"]

  if metastage == "metatrain":  
    path = repo_path + "sample_data/pretrainPanCan/" + title
    write_datasets_metatrain(path, X_train, X_holdout,ytime_train, ystatus_train,ytime_holdout,ystatus_holdout)
  else:
    path = repo_path + "sample_data/finetuneTarget/" + title
    write_datasets_metatest(path, X_train, X_holdout, ytime_train, ystatus_train, ytime_holdout, ystatus_holdout)


def write_datasets_metatrain(path, X_train, X_holdout, ytime_train, ystatus_train, ytime_holdout, ystatus_holdout):
  X_train.to_csv(path+"_feature_train.csv",index=False)
  ytime_train.to_csv(path+"_ytime_train.csv",index=False)
  ystatus_train.to_csv(path+"_ystatus_train.csv",index=False)

  X_holdout.to_csv(path+"_feature_val.csv",index=False) ##For the metatrain stage the holdout is what is used for validation
  ytime_holdout.to_csv(path+"_ytime_val.csv",index=False)
  ystatus_holdout.to_csv(path+"_ystatus_val.csv",index=False)

def write_datasets_metatest(path, X_train, X_holdout, ytime_train, ystatus_train, ytime_holdout, ystatus_holdout):
  X_train.to_csv(path+"_feature_train.csv",index=False)
  ytime_train.to_csv(path+"_ytime_train.csv",index=False)
  ystatus_train.to_csv(path+"_ystatus_train.csv",index=False)

  X_holdout.to_csv(path+"_feature_holdout.csv",index=False)
  ytime_holdout.to_csv(path+"_ytime_holdout.csv",index=False)
  ystatus_holdout.to_csv(path+"_ystatus_holdout.csv",index=False)

#Test Cancer - All except for GBM, LGG, LUAD, LUSC, HNSC, MESO
exclude_list = ["GBM", "LGG", "LUAD", "LUSC", "HNSC", "MESO"]
meta_train = protein_expression_tcga.drop(protein_expression_tcga.index[protein_expression_tcga['cancer_type'].isin(exclude_list)])

#Target Cancer types -  20 samples each of (GBB, LGG, LUAD, LUSC, HNSC) , All samples of MESO
meta_test_GBM = protein_expression_tcga.drop(protein_expression_tcga.index[protein_expression_tcga['cancer_type'] != "GBM"]).sample(n = 20)
meta_test_LGG = protein_expression_tcga.drop(protein_expression_tcga.index[protein_expression_tcga['cancer_type'] != "LGG"]).sample(n = 20)
meta_test_LUAD = protein_expression_tcga.drop(protein_expression_tcga.index[protein_expression_tcga['cancer_type'] != "LUAD"]).sample(n = 20)
meta_test_LUSC = protein_expression_tcga.drop(protein_expression_tcga.index[protein_expression_tcga['cancer_type'] != "LUSC"]).sample(n = 20)
meta_test_HNSC = protein_expression_tcga.drop(protein_expression_tcga.index[protein_expression_tcga['cancer_type'] != "HNSC"]).sample(n = 20)

meta_test_MESO = protein_expression_tcga.drop(protein_expression_tcga.index[protein_expression_tcga['cancer_type'] != "MESO"])

split_train_test(meta_train, "protein_pancan_v1", "metatrain")
split_train_test(meta_test_MESO, "MESO", "metatest")
split_train_test(meta_test_GBM, "GBM", "metatest")
split_train_test(meta_test_LGG, "LGG", "metatest")
split_train_test(meta_test_LUAD, "LUAD", "metatest")
split_train_test(meta_test_LUSC, "LUSC", "metatest")
split_train_test(meta_test_HNSC, "HNSC", "metatest")