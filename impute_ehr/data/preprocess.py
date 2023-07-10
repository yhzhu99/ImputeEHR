import copy
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def export_patient_list_pickle(df, val_ratio=0.2):
    seed = 42

    # Group the dataframe by patient ID
    grouped = df.groupby('PatientID')

    # Get the patient IDs and outcomes
    patients = np.array(list(grouped.groups.keys()))
    patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in patients])

    # Get the train_val/test patient IDs
    train_patients, val_patients = train_test_split(patients, test_size=20/100, random_state=seed, stratify=patients_outcome)
    
    # Get the train/val dataframes
    train_df = df[df['PatientID'].isin(train_patients)]
    val_df = df[df['PatientID'].isin(val_patients)]

    # Group by PatientID, Convert dataframes to nested lists
    train_list = train_df.groupby('PatientID').apply(lambda x: x.values.tolist()).tolist()
    val_list = val_df.groupby('PatientID').apply(lambda x: x.values.tolist()).tolist()

    # only the index 6: are required
    train_list = [[x[6:] for x in patient] for patient in train_list]
    val_list = [[x[6:] for x in patient] for patient in val_list]

    # export to pickle
    pd.to_pickle(train_list, './datasets/impute_train_x.pkl')
    pd.to_pickle(val_list, './datasets/impute_val_x.pkl')

def flatten_to_matrix(data):
    """
    Flatten the nested list to matrix
    """
    return np.array([x for patient in data for x in patient])