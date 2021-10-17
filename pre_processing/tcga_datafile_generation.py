## Creating a dataframe with all TCGA OMICS (Protein/MicroRNA) Expression Data, Survival Time and Survival Status ##

import os
import pandas as pd
import numpy as np
import mat73
from scipy import io


def extract_features(cons_features):
  feature_list = []
  for feature in cons_features['all_features']:
    feature_list.append(feature[0][0])

  feature_list = [x.upper() for x in feature_list] #Converting all the features to upper case
  return feature_list

def extract_patient_time_status(cons_p_info):
  patient_list = []
  time_list = []
  status_list = []
  for patient in cons_p_info['patient_info']:
    patient_list.append(patient[2][0])

    if patient[5].size != 0: #Checking if the cell is not empty
      if patient[5][0] == 'alive':
        status_list.append(1)
      else:
        status_list.append(0)
    else:
      status_list.append(np.NaN)

    if patient[7].size != 0: #Checking if the cell is not empty
      time_list.append(patient[7][0])
    else:
      time_list.append(np.NaN)

  return patient_list, time_list, status_list

def build_df(cancer_type, patient_list, time_list, status_list, feature_list, cons_data):
  time_df = pd.DataFrame(time_list, index =patient_list, columns = ["time"])
  status_df = pd.DataFrame(status_list, index =patient_list, columns = ["status"])

  feature_sample_matrix = cons_data['all_data']
  feature_df = pd.DataFrame(feature_sample_matrix, index =feature_list, columns = patient_list)
  feature_df = feature_df.T

  combined_df = pd.concat([feature_df, time_df, status_df ], axis=1)

  return combined_df

def main(datatype_extension):
  all_dfs = []
  for folder in os.listdir():
    if folder.startswith(datatype_extension):
      cancer_type = folder.lstrip(datatype_extension)

      try:
        cons_features = io.loadmat(folder+"/consolidated_features.mat")
        cons_data = mat73.loadmat(folder+"/consolidated_data.mat")
        cons_p_info = io.loadmat(folder+"/consolidated_patient_info.mat")

        feature_list = extract_features(cons_features)
        patient_list, time_list, status_list = extract_patient_time_status(cons_p_info)

        combined_df = build_df(cancer_type, patient_list, time_list, status_list, feature_list, cons_data)
        combined_df['cancer_type'] = cancer_type
        all_dfs.append(combined_df)
        print("Success for ", cancer_type)

      except:
        print("Failed for ", cancer_type)

  master_df = pd.DataFrame() #The braces are to prevent error: The reason behind the error is the first argument of a method is the class instance.
  master_df = master_df.append(all_dfs, sort=False)

  return master_df

######## PIPELINE EXECUTION ########

#### MICRORNA EXPRESSION DATA ####
# %cd /content/gdrive/My Drive/metalearning_survival/tcga_data/MicroRNA_Expression
microrna_expression_tcga = main("MicroRNA_")
microrna_expression_tcga.to_csv("tcga_microrna_df.csv")

#### PROTEIN EXPRESSION DATA ####
# %cd /content/gdrive/My Drive/metalearning_survival/tcga_data/Protein_RPPA
protein_expression_tcga = main("Protein_")
protein_expression_tcga.to_csv("tcga_protein_df.csv")
