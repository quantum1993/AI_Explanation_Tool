import pandas as pd
pd.options.display.max_columns = 50

# df = pd.DataFrame()
# df['a'] = [1, 2, 3]
#
# mapping = {1: ['a', 0], 2: ['b', 0], 3: ['c', 0]}
#
# # df[['T', 'K']] = df['a'].apply(lambda x: mapping[x])
# print(df['a'].apply(lambda x: pd.Series(data=mapping[x], index=['T', 'K'])))

a = ['CHAS_cat', 'dependence_CHAS_cat', 'RAD', 'dependence_RAD', 'AGE',
       'dependence_AGE', 'ID', 'NOX', 'dependence_NOX', 'CRIM',
       'dependence_CRIM', 'PTRATIO', 'dependence_PTRATIO', 'RM',
       'dependence_RM', 'CHAS', 'B', 'dependence_B', 'INDUS',
       'dependence_INDUS', 'TAX', 'dependence_TAX', 'DIS', 'dependence_DIS',
       'LSTAT', 'dependence_LSTAT', 'ZN', 'dependence_ZN', 'PRICE', 'y_pred',
       'decision_x', 'decision_y', 'index', 'status', 'color', 'color_select']
b = ['ID', 'CHAS_cat', 'dependence_CHAS_cat', 'RAD', 'dependence_RAD', 'AGE',
       'dependence_AGE', 'NOX', 'dependence_NOX', 'CRIM', 'dependence_CRIM',
       'PTRATIO', 'dependence_PTRATIO', 'RM', 'dependence_RM', 'CHAS', 'B',
       'dependence_B', 'INDUS', 'dependence_INDUS', 'TAX', 'dependence_TAX',
       'DIS', 'dependence_DIS', 'LSTAT', 'dependence_LSTAT', 'ZN',
       'dependence_ZN', 'PRICE', 'y_pred', 'decision_x', 'decision_y', 'index',
       'status', 'color', 'color_select']

print(set(a) - set(b))
print(set(b) - set(a))

