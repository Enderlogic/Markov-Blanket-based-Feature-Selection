import os
import random
import urllib
from pathlib import Path

import pandas
import rpy2.robjects as ro
from bnsl import add_missing, from_bnlearn
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

base, bnlearn = importr('base'), importr('bnlearn')

model_list = ['asia', 'alarm', 'insurance', 'hailfinder']
datasize = 10000
noise_list = ['MCAR', 'MAR', 'MNAR']
error_rate_list = [0.1, 0.3, 0.5]


# create missing mechanism
def miss_mechanism(dag, noise='MAR', rom=0.5):
    '''

    :param dag: true DAG
    :param noise: type of missing mechanism
    :param rom: ratio of missing variables
    :return: missing mechanism
    '''
    if type(dag) is str:
        dag = from_bnlearn(dag)
    elif type(dag) is not dict:
        raise Exception('The format of input DAG is invalid.')
    varnames = list(dag.keys())
    cause_dict = {}
    vars_miss = random.sample(varnames, round(len(varnames) * rom))
    vars_comp = [v for v in varnames if v not in vars_miss]
    if noise == 'MCAR':
        for var in vars_miss:
            cause_dict[var] = []
    elif noise == 'MAR':
        for var in vars_miss:
            cause_dict[var] = random.sample(vars_comp, 1)
    elif noise == 'MNAR':
        for var in vars_miss:
            cause_dict[var] = random.sample([v for v in vars_miss if v != var], 1)
    else:
        raise Exception('noise ' + noise + ' is undefined.')
    return cause_dict


for model in model_list:
    # load complete model
    model_path = 'model/' + model + '.rds'
    if not os.path.isfile(model_path):
        Path(os.path.dirname(model_path)).mkdir(parents=True, exist_ok=True)
        if model != 'mehra':
            urllib.request.urlretrieve('https://www.bnlearn.com/bnrepository/' + model + '/' + model + '.rds',
                                       model_path)
        else:
            urllib.request.urlretrieve('https://www.bnlearn.com/bnrepository/mehra/mehra-complete.rds', model_path)
    data_path = 'data/' + model + '/Clean.csv'
    dag_true = base.readRDS(model_path)

    if not os.path.isfile(data_path):
        data = bnlearn.rbn(dag_true, datasize)
        with localconverter(ro.default_converter + pandas2ri.converter):
            data = ro.conversion.rpy2py(data)
        data = data[random.sample(list(data), data.shape[1])]
        data.index = data.index.astype('int')
        Path(os.path.dirname(data_path)).mkdir(parents=True, exist_ok=True)
        data.to_csv(data_path, index=False)
    else:
        data = pandas.read_csv(data_path)
    for noise in noise_list:
        for error_rate in error_rate_list:
            data_path = 'data/' + str(model) + '/' + noise + '_' + str(error_rate) + '.csv'
            if not os.path.isfile(data_path):
                cause_dict = miss_mechanism(bnlearn.modelstring(dag_true)[0], noise)
                data_missing = add_missing(data, cause_dict, m_max=error_rate)
                Path(os.path.dirname(data_path)).mkdir(parents=True, exist_ok=True)
                data_missing.to_csv(data_path, index=False)
