import pandas as pd
pd.options.display.max_columns = 50
# df = pd.read_csv('train.csv')
# print(df['CRIM'][-1])
# print(df.describe())

df = pd.DataFrame()
df['DDD'] = ['A', 'B']
df['LLL'] = [1, 2]
print(df[df['DDD'] == 'B']['LLL'])
print(df[df['DDD'] == 'B']['LLL'][0])
print(type(df[df['DDD'] == 'B']['LLL']))


# step_dict = dict()
# step_dict['CRIM'] = 0.00001
# step_dict['ZN'] = 0.5
#
# print(step_dict.values())