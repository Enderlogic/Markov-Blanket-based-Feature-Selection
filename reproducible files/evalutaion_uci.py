import os.path

import pandas
from rpy2.robjects.packages import importr

from accessories import rmse, pfc

base, bnlearn = importr('base'), importr('bnlearn')

model_list = ['car', 'mushroom', 'game', 'iris', 'breast', 'wine']
missing_type_list = ['MCAR', 'MAR', 'MNAR']
missing_rate_list = [0.1, 0.3, 0.5]
algorithm_list = ['Mean', 'Mode', 'KNN', 'GAIN', 'softImpute', 'MF', 'MBMF']
result_path = 'results_uci.csv'
columns = ['model', 'datasize', 'missingtype', 'missingrate', 'algorithm', 'rmse', 'F1']
result = pandas.DataFrame(columns=columns)
result_list = result.values.tolist()

for model in model_list:
    data = pandas.read_csv('../data/' + model + '/Complete.csv')
    for missing_type in missing_type_list:
        for missing_rate in missing_rate_list:
            data_missing = pandas.read_csv('../data/' + model + '/' + missing_type + '_' + str(missing_rate) + '.csv')
            for algorithm in algorithm_list:
                data_path = '../imputed_data/' + model + '/' + missing_type + '_' + str(missing_rate) + '_' + algorithm + '.csv'
                if os.path.isfile(data_path):
                    data_impute = pandas.read_csv(data_path)
                    if model in ['car', 'game', 'mushroom']:
                        result_list.append([model, missing_type, missing_rate, algorithm, pfc(data, data_impute, data_missing)])
                    else:
                        result_list.append([model, missing_type, missing_rate, algorithm, rmse(data, data_impute, data_missing)])
                    print(result_list[-1])

pandas.DataFrame(result_list, columns=columns).to_csv(result_path, index=False)