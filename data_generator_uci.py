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

model_list = ['car', 'game', 'mushroom']
noise_list = ['MCAR', 'MAR', 'MNAR']
error_rate_list = [0.1, 0.3, 0.5]


# create missing mechanism
def miss_mechanism(data, noise='MAR', rom=0.5):
    '''

    :param dag: true DAG
    :param noise: type of missing mechanism
    :param rom: ratio of missing variables
    :return: missing mechanism
    '''
    varnames = list(data.columns)
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
    data_path = 'data/' + model + '/Clean.csv'
    data = pandas.read_csv(data_path, dtype='category')
    for noise in noise_list:
        for error_rate in error_rate_list:
            data_path = 'data/' + str(model) + '/' + noise + '_' + str(error_rate) + '.csv'
            if not os.path.isfile(data_path):
                cause_dict = miss_mechanism(data, noise)
                data_missing = add_missing(data, cause_dict, m_max=error_rate)
                Path(os.path.dirname(data_path)).mkdir(parents=True, exist_ok=True)
                data_missing.to_csv(data_path, index=False)
