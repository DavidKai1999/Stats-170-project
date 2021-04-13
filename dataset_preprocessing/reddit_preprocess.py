import pandas as pd

file_name = 'FactCheckData_clean_mini.json'


pd.set_option('display.max_columns', None)

data = pd.read_json(file_name,encoding='utf-8')

# print(data[~data['rating'].str.contains('True')])

FalseData = data[~data['rating'].str.contains('True').reset_index(drop = True)]
TrueData = data[data['rating'].str.contains('True').reset_index(drop = True)]

TrueData.drop('rating',inplace=True,axis = 1)
TrueData['rating'] = 0
FalseData.drop('rating',inplace=True,axis = 1)
FalseData['rating'] = 1


data = pd.concat([TrueData, FalseData]).reset_index(drop = True)
# print('all text',type(data),data)#,all_text
