import os

import numpy
import pandas
import rpy2.robjects as ro
from sklearn.impute import KNNImputer, SimpleImputer

from GAIN.gain import gain
from accessories import rmse
from softimpute import softimpute, cv_softimpute

ro.r('''source('../missForest.r')''')
missForest = ro.globalenv['missForest']

model_list = ['ecoli70', 'magic-irri', 'arth150']
datasize_list = [500, 1000, 2000, 3000]
missing_type_list = ['MCAR', 'MAR', 'MNAR']
missing_rate_list = [0.1, 0.3, 0.5]
algorithm_list = ['Mean', 'KNN', 'GAIN', 'softImpute', 'MF', 'MBMF']
gain_parameters = {'batch_size': 64,
                   'hint_rate': 0.9,
                   'alpha': 10,
                   'iterations': 10000}

for model in model_list:
    # load complete model
    data = pandas.read_csv('../data/' + model + '/Complete.csv')
    for missing_type in missing_type_list:
        for missing_rate in missing_rate_list:
            # load missing model
            data_missing = pandas.read_csv('../data/' + model + '/' + missing_type + '_' + str(missing_rate) + '.csv')
            for datasize in datasize_list:
                for algorithm in algorithm_list:
                    data_path = '../imputed_data/' + model + '/' + missing_type + '_' + str(missing_rate) + '_' + str(
                        datasize) + '_' + algorithm + '.csv'
                    if algorithm == 'GAIN':
                        data_imputed = pandas.DataFrame(gain(data_missing.head(datasize).to_numpy(), gain_parameters),
                                                        columns=data.columns)
                    elif algorithm == 'KNN':
                        imputer = KNNImputer()
                        data_imputed = pandas.DataFrame(imputer.fit_transform(data_missing.head(datasize)),
                                                        columns=list(data.columns))
                    elif algorithm == 'Mean':
                        imputer = SimpleImputer(missing_values=numpy.nan, strategy='mean')
                        data_imputed = pandas.DataFrame(imputer.fit_transform(data_missing.head(datasize)),
                                                        columns=list(data.columns))
                    elif algorithm == 'MF':
                        data_imputed = ro.conversion.rpy2py(
                            missForest(ro.conversion.py2rpy(data_missing.head(datasize)), fs='None')[0]).reset_index()
                    elif algorithm == 'MBMF':
                        data_imputed = ro.conversion.rpy2py(
                            missForest(ro.conversion.py2rpy(data_missing.head(datasize)), fs='mbfs')[0]).reset_index()
                    elif algorithm == 'softImpute':
                        cv_error, grid_lambda = cv_softimpute(data_missing.head(datasize).to_numpy(), grid_len=5)
                        lbda = grid_lambda[numpy.argmin(cv_error)]
                        data_imputed = pandas.DataFrame(softimpute(data_missing.head(datasize).to_numpy(), lbda)[1],
                                                        columns=data.columns)
                    else:
                        raise Exception('Undefined method (' + algorithm + ').')
                    os.makedirs(os.path.dirname(data_path), exist_ok=True)
                    pandas.DataFrame(data_imputed, columns=list(data.columns)).to_csv(data_path, index=False)
                    print(model, missing_type, missing_rate, datasize, algorithm)
                    print('root mean square error:',
                          rmse(data.head(datasize), data_imputed, data_missing.head(datasize)))
