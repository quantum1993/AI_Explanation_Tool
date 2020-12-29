from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import os
import pickle
import shap
from .util import decision_value_core, get_importance_list, save_pickle


def data_prepare():
    save_path = 'data'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    X, Y = load_boston(return_X_y=True)
    feature_names = load_boston().feature_names
    df = pd.DataFrame(data=X, columns=feature_names)
    df['PRICE'] = Y
    df['ID'] = ['S'+str(x) for x in list(range(len(X)))]
    df.to_csv(os.path.join(save_path, 'train.csv'), index=False)

    id_cols = df[['ID']]
    y = df[['PRICE']]
    col_names = [x for x in df.columns if x not in ['ID', 'PRICE']]
    x = df[col_names]
    model = GradientBoostingRegressor(max_depth=10)
    model.fit(x.values, y.values.flatten())
    with open(os.path.join(save_path, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)
    df_shap = pd.DataFrame(data=shap_values, columns=col_names)
    importance_list = get_importance_list(df_shap).Features.to_list()
    decision_values = decision_value_core(df_shap, explainer.expected_value,
                                          idx_names=df['ID'].to_list(),
                                          importance_list=importance_list)
    save_pickle(explainer, os.path.join(save_path, 'explainer.pkl'))
    df_shap.to_csv(os.path.join(save_path, 'shap_values.csv'), index=False)
    decision_values.to_csv(os.path.join(save_path, 'decision_values.csv'), index=False)


