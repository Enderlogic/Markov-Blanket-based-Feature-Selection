import pandas
from bnsl import from_bnlearn, f1, ges
from rpy2.robjects.packages import importr

from accessories import rmse

base, bnlearn = importr('base'), importr('bnlearn')

model_list = ['ecoli70', 'magic-irri', 'arth150']
missing_type_list = ['MCAR', 'MAR', 'MNAR']
missing_rate_list = [0.1, 0.3, 0.5]
datasize_list = [500, 1000, 2000, 3000]
algorithm_list = ['Mean', 'KNN', 'GAIN', 'softImpute', 'MF', 'MBMF']
result_path = '../results_bn.csv'
columns = ['model', 'datasize', 'missingtype', 'missingrate', 'algorithm', 'rmse', 'F1']
result = pandas.DataFrame(columns=columns)
result_list = result.values.tolist()

for model in model_list:
    dag = from_bnlearn(bnlearn.modelstring(base.readRDS('../model/' + model + '.rds'))[0])
    for datasize in datasize_list:
        data = pandas.read_csv('../data/' + model + '/Complete.csv')
        result_list.append([model, datasize, 'Complete', 0, None, 0, f1(dag, ges(data.head(datasize)))])
        print(result_list[-1])
        for missing_type in missing_type_list:
            for missing_rate in missing_rate_list:
                data_missing = pandas.read_csv(
                    '../data/' + model + '/' + missing_type + '_' + str(missing_rate) + '.csv')
                for algorithm in algorithm_list:
                    data_impute = pandas.read_csv(
                        '../imputed_data/' + model + '/' + missing_type + '_' + str(missing_rate) + '_' + str(
                            datasize) + '_' + algorithm + '.csv')
                    result_list.append([model, datasize, missing_type, missing_rate, algorithm,
                                        rmse(data.head(datasize), data_impute, data_missing.head(datasize)),
                                        f1(dag, ges(data_impute))])
                    print(result_list[-1])

pandas.DataFrame(result_list, columns=columns).to_csv(result_path, index=False)
