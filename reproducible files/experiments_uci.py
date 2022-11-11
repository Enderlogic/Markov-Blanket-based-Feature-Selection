import os
from pathlib import Path

import numpy
import pandas
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from softimpute import softimpute, cv_softimpute
from GAIN.gain import gain
from rpy2.robjects.conversion import localconverter

ro.r('''source('../missForest.r')''')
missForest = ro.globalenv['missForest']


def find_category_mappings(df, variable):
    return {k: i for i, k in enumerate(df[variable].dropna().unique(), 0)}


def integer_encode(df, variable, ordinal_mapping):
    df[variable] = df[variable].map(ordinal_mapping)


def imputation(df1, cols):
    mappin = dict()
    mm = MinMaxScaler()
    df = df1.copy()
    # Encoding dict &amp; Removing nan
    # mappin = dict()
    for variable in cols:
        mappings = find_category_mappings(df, variable)
        mappin[variable] = mappings

    # Apply mapping
    for variable in cols:
        integer_encode(df, variable, mappin[variable])

        # Minmaxscaler and KNN imputation
    df = df.astype('float')
    sca = mm.fit_transform(df)
    knn_imputer = KNNImputer()
    knn = knn_imputer.fit_transform(sca)
    df.iloc[:, :] = mm.inverse_transform(knn)
    for i in df.columns:
        df[i] = round(df[i]).astype('int')

    # Inverse transform
    for i in cols:
        inv_map = {v: k for k, v in mappin[i].items()}
        df[i] = df[i].map(inv_map)
    return df


gain_parameters = {'batch_size': 64,
                   'hint_rate': 0.9,
                   'alpha': 10,
                   'iterations': 10000}

model_list = ['iris', 'breast', 'wine', 'car', 'game', 'mushroom']
missing_type_list = ['MCAR', 'MAR', 'MNAR']
algorithm_list = ['Mean', 'Mode', 'KNN', 'GAIN', 'MF', 'MBMF', 'softImpute']
missing_rate_list = [0.1, 0.3, 0.5]

for model in model_list:
    for missing_type in missing_type_list:
        for missing_rate in missing_rate_list:
            # load missing model
            data_path = '../data/' + model + '/' + missing_type + '_' + str(missing_rate) + '.csv'
            data_missing = pandas.read_csv(data_path, dtype='category') if model in ['car', 'game',
                                                                                     'mushroom'] else pandas.read_csv(
                data_path)
            with localconverter(ro.default_converter + pandas2ri.converter):
                data_missing_r = ro.conversion.py2rpy(data_missing)
            for algorithm in algorithm_list:
                data_path = '../imputed_data/' + str(model) + '/' + missing_type + '_' + str(
                    missing_rate) + '_' + algorithm + '.csv'
                if algorithm == 'GAIN':
                    if model in ['car', 'game', 'mushroom']:
                        continue
                    else:
                        data_imputed = pandas.DataFrame(gain(data_missing.to_numpy(), gain_parameters),
                                                        columns=data_missing.columns)
                elif algorithm == 'KNN':
                    if model in ['car', 'game', 'mushroom']:
                        data_imputed = imputation(data_missing, data_missing.columns.tolist())
                    else:
                        imputer = KNNImputer()
                        data_imputed = pandas.DataFrame(imputer.fit_transform(data_missing),
                                                        columns=list(data_missing.columns))
                elif algorithm == 'Mode':
                    if model in ['car', 'game', 'mushroom']:
                        imputer = SimpleImputer(missing_values=numpy.nan, strategy='most_frequent')
                        data_imputed = pandas.DataFrame(imputer.fit_transform(data_missing),
                                                        columns=list(data_missing.columns))
                    else:
                        continue
                elif algorithm == 'Mean':
                    if model in ['car', 'game', 'mushroom']:
                        continue
                    else:
                        imputer = SimpleImputer(missing_values=numpy.nan, strategy='mean')
                        data_imputed = pandas.DataFrame(imputer.fit_transform(data_missing),
                                                        columns=list(data_missing.columns))
                elif algorithm == 'MF':
                    data_imputed = missForest(data_missing_r, fs='None')[0]
                    with localconverter(ro.default_converter + pandas2ri.converter):
                        data_imputed = ro.conversion.rpy2py(data_imputed)

                elif algorithm == 'MBMF':
                    data_imputed = missForest(data_missing_r, fs='mbfs')[0]
                    with localconverter(ro.default_converter + pandas2ri.converter):
                        data_imputed = ro.conversion.rpy2py(data_imputed)
                elif algorithm == 'SoftImpute':
                    if model in ['car', 'game', 'mushroom']:
                        continue
                    else:
                        cv_error, grid_lambda = cv_softimpute(data_missing.to_numpy(), grid_len=5)
                        lbda = grid_lambda[numpy.argmin(cv_error)]
                        data_imputed = pandas.DataFrame(softimpute(data_missing.to_numpy(), lbda)[1],
                                                        columns=data_missing.columns)
                else:
                    raise Exception('Undefined method ' + algorithm + '.')
                Path(os.path.dirname(data_path)).mkdir(parents=True, exist_ok=True)
                data_imputed.to_csv(data_path, index=False)
                print(model, missing_type, missing_rate, algorithm)
