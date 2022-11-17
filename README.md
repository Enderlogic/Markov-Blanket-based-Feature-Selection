# Markov Blanket-based Feature Selection

This is an example code to demonstrate a Markov Blanket-based Feature Selection (MBFS) algorithm for imputation. We
incorporate this idea with the MissForest imputation algorithm. 

## Command inputs:

- data_type: synthetic or real-world
- data_name: ecoli70 (synthetic) or breast (real-world)
- data_size: data size (for synthetic data only)
- missing_type: type of missingness (MCAR, MAR or MNAR)
- error_rate: maximum error rate for partially observed variables
- ratio_of_partially_observed_variables: proportion of partially observed variables
- feature_selection: feature selection approach (None or mbfs)

## Example:

```angular2html
$ python3 main.py --data_type 'synthetic' --data_name 'ecoli70' --data_size 1000 --noise_type 'MCAR' --error_rate 0.3 --ratio_of_partially_observed_variables 0.5 --feature_selection 'mbfs'
```

## Output:
- imputed_data: imputed data set


## Reproducibility:

In order to reproduce our results, please do the following steps:

1. Download the data sets from this [repository][1] and replace the original data folder.
2. Execute the experiments_bn.py/experiments_uci.py in the reproducible files folder to generate imputed data sets for synthetic/real-world experiments. The imputed data will be saved in the imputed_data folder
3. Execute the evaluation_bn.py/evaluation_uci.py in the reproducible files folder to produce the evaluation results for synthetic/real-world experiments. The final results will be saved as results_bn.csv/results_uci.csv in the main folder.
4. Execute the plot_bn.R in the reproducible files folder to plot the results of synthetic experiments. The plot will be saved in the main folder.

[1]: https://github.com/Enderlogic/MBMF_data_repository "data repository"
