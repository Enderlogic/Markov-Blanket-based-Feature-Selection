# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Main function for UCI letter and spam datasets.
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
import rpy2.robjects as ro
from sklearn.impute import KNNImputer

pandas2ri.activate()
missForest = importr('missForest')

from data_loader import data_loader
from gain import gain
from utils import rmse_loss

# np.random.seed(990806)
data_name = 'spam'
miss_rate = 0.2

gain_parameters = {'batch_size': 64,
                 'hint_rate': 0.9,
                 'alpha': 100,
                 'iterations': 10000}

# Load data and introduce missingness
ori_data_x, miss_data_x, data_m = data_loader(data_name, miss_rate)
# ori_data_x = pandas.read_csv(data_name + '.csv', header=None).to_numpy()
# miss_data_x = pandas.read_csv(data_name + '_m.csv', header=None).to_numpy()
# data_m = 1 - np.isnan(miss_data_x).astype(int)

# pandas.DataFrame(ori_data_x).to_csv('letter.csv', header=False, index=False)
# pandas.DataFrame(miss_data_x).to_csv('letter_m.csv', header=False, index=False)

# Impute missing data
imputed_data_mf = missForest.missForest(pandas.DataFrame(miss_data_x))[0]
imputed_data_mf = ro.conversion.rpy2py(imputed_data_mf).to_numpy()
# imputed_data_mf = pandas.read_csv(data_name + '_mf.csv').to_numpy()
rmse_mf = rmse_loss(ori_data_x, imputed_data_mf, data_m)
print('MissForest RMSE Performance: ' + str(np.round(rmse_mf, 4)))

imputer = KNNImputer()
imputed_data_knn = imputer.fit_transform(miss_data_x)
rmse_gain = rmse_loss(ori_data_x, imputed_data_knn, data_m)
print('KNN RMSE Performance: ' + str(np.round(rmse_gain, 4)))


imputed_data_gain = gain(miss_data_x, gain_parameters)
rmse_gain = rmse_loss(ori_data_x, imputed_data_gain, data_m)
print('GAIN RMSE Performance: ' + str(np.round(rmse_gain, 4)))
