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
$ python3 main.py --data_type 'synthetic' --data_name: 'ecoli70'
--data_size 1000 --noise_type 'MCAR' --error_rate 0.3
--ratio_of_partially_observed_variables 0.5 --feature_selection 'mbfs'
```

## Output:
- imputed_data: imputed data set
