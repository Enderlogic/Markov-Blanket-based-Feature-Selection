import argparse
import random

import pandas
import rpy2.robjects as ro
from bnsl import ges, f1, add_missing, from_bnlearn
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

pandas2ri.activate()

from accessories import rmse

base, bnlearn = importr('base'), importr('bnlearn')
ro.r('''source('missForest.r')''')
missForest = ro.globalenv['missForest']


# create missing mechanism
def miss_mechanism(varnames, missingtype='MAR', rom=0.5):
    '''

    :param dag: true DAG
    :param noise: type of missing mechanism
    :param rom: ratio of missing variables
    :return: missing mechanism
    '''
    cause_dict = {}
    vars_miss = random.sample(varnames, round(len(varnames) * rom))
    vars_comp = [v for v in varnames if v not in vars_miss]
    if missingtype == 'MCAR':
        for var in vars_miss:
            cause_dict[var] = []
    elif missingtype == 'MAR':
        for var in vars_miss:
            cause_dict[var] = random.sample(vars_comp, 1)
    elif missingtype == 'MNAR':
        for var in vars_miss:
            cause_dict[var] = random.sample([v for v in vars_miss if v != var], 1)
    else:
        raise Exception('noise ' + missingtype + ' is undefined.')
    return cause_dict


def main(args):
    '''

    :param args
        data_type: synthetic or real-world
        data_name: the name of data, for example, 'ecoli70' (synthetic) and 'breast' (real-world)
        data_size: number of instance (for synthetic only)
        missing_type: type of missingness ('MCAR', 'MAR' or 'MNAR')
        ratio_of_partially_observed_variables: proportion of missing values
        error_rate: maximum error rate
        feature_selection: feature selection approach ('None' or 'mbfs')
    :return:
        data_imputed: imputed data
    '''
    if args.data_type == 'synthetic':
        model = base.readRDS('model/' + args.data_name + '.rds')
        data_clean = ro.conversion.rpy2py(bnlearn.rbn(model, args.data_size))
    elif args.data_type == 'real-world':
        data_clean = pandas.read_csv('data/' + args.data_name + '.csv')
    else:
        raise Exception('Unknown data_type:', args.data_type, '. Should be either \'synthetic\' or \'real world\' ')

    data_missing = add_missing(data_clean, miss_mechanism(list(data_clean.columns), args.missing_type,
                                                          args.ratio_of_partially_observed_variables),
                               m_max=args.error_rate)
    data_imputed = ro.conversion.rpy2py(missForest(ro.conversion.py2rpy(data_missing), fs=args.feature_selection)[0])

    print('data:', args.data_name)
    print('data size:', args.data_size)
    print('missing type:', args.missing_type)
    print('error rate', args.error_rate)
    print('feature selection:', args.feature_selection)
    print('RMSE:', rmse(data_clean, data_imputed, data_missing))
    if args.data_type == 'synthetic':
        print('F1:', f1(bnlearn.modelstring(model)[0], ges(data_imputed)))

    return data_imputed


if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_type',
        choices=['synthetic', 'real-world'],
        default='synthetic',
        type=str)
    parser.add_argument(
        '--data_name',
        choices=['ecoli70', 'breast'],
        default='ecoli70',
        type=str)
    parser.add_argument(
        '--data_size',
        help='number of instance for synthetic data',
        default=1000,
        type=int)
    parser.add_argument(
        '--missing_type',
        choices=['MCAR', 'MAR', 'MNAR'],
        default='MCAR',
        type=str)
    parser.add_argument(
        '--error_rate',
        help='maximum error rate',
        default=0.3,
        type=float)
    parser.add_argument(
        '--ratio_of_partially_observed_variables',
        help='ratio of partially observed varaibles',
        default=0.5,
        type=float)
    parser.add_argument(
        '--feature_selection',
        choices=['None', 'mbfs'],
        default='mbfs',
        type=str)
    args = parser.parse_args()

    # Calls main function
    imputed_data = main(args)
