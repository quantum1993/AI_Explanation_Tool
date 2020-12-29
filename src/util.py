import pandas as pd
import numpy as np
import pickle
import yaml
from os.path import join


def get_importance_list(df_shap, save_path=None):
    assert str(type(df_shap)) == "<class 'pandas.core.frame.DataFrame'>", "type of df_shap must be pandas DataFrame"
    feature_importance = np.sum(np.abs(df_shap.values), axis=0)
    df = pd.DataFrame()
    df["Features"] = list(df_shap.columns)
    df["Importance"] = feature_importance
    df = df.sort_values("Importance", ascending=False).reset_index(drop=True)
    if save_path is not None:
        df.to_csv(save_path, index=False)
    return df


def reorder_shap_values(df_shap, importance_list, idx_names):
    df_shap = df_shap.copy()
    df_shap = df_shap[importance_list]
    tmp = pd.DataFrame(data=df_shap.values.T, columns=idx_names)
    tmp["parameters"] = importance_list
    return tmp


def get_decision_values(shap_values_or, expected_value):
    decision_values = pd.DataFrame()
    base_data = [expected_value] * (len(shap_values_or.columns) - 1)
    base_data.append("Base")
    base_df = pd.DataFrame(data=np.array([base_data]),
                           columns=shap_values_or.columns)
    shap_values_or = shap_values_or.append(base_df, ignore_index=True)
    for col in shap_values_or.columns:
        if col == "parameters":
            continue
        shap_values_or[col] = shap_values_or[col].astype('float64')
        decision_values[col] = shap_values_or.loc[::-1, col].cumsum()[::-1]
    decision_values["parameters"] = shap_values_or["parameters"]
    decision_values["y_pos"] = list(decision_values.index.values)[::-1]
    return decision_values


def decision_value_core(df_shap, expected_value, idx_names, importance_list):
    shap_values_or = reorder_shap_values(df_shap,
                                         importance_list=importance_list,
                                         idx_names=idx_names)
    decision_values = get_decision_values(shap_values_or, expected_value)
    return decision_values


def get_dependence_values(col, shap_values):
    return shap_values[col].values


def save_pickle(file, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(file, f)
    return None


def load_pickle(save_path):
    with open(save_path, 'rb') as f:
        file = pickle.load(f)
    return file


def load_yaml(config_path):
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config