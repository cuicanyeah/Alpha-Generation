# Stock-Prediction-System

The repository is the official implementation of the stock prediction system.

## Environment:

We run our experiments on Intel(R) Xeon(R) W-2133 CPU @ 3.60GHz. Mutiple CPUs are recommended for running alpha initializations in parallel. GPU is not necessary but useful in running deep learning models to be implemented later. 

## Requirements:

Install Bazel version of 4.2.1 from the below link, and g++>=9 is required. Python version 3.8.8 is used in running deep learning models.
[https://docs.bazel.build/versions/main/install-ubuntu.html](https://docs.bazel.build/versions/main/install-ubuntu.html)

## Data:

The data folder contains ten years of U.S. stocks from 2013 to 2022 downloaded from Eikon. You can complement the data with relational data from the `data` folder in the below GitHub repository. 
[https://github.com/fulifeng/Temporal_Relational_Stock_Ranking](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking)

## Preprocessing commands:

Sample command to generate a dataset for AlphaEvolve as inputs using raw data (i.e., open/high/low/close/volume).
```
python preprocess/generate_datasets_for_newly_created_datasets.py -data_dir processed_data -path raw_data/eikon_data -market ALL -num_train_examples 1 -num_valid_examples 471 -start_date 2017-1-2 -end_date 2019-1-2
```

Run an alpha evaluation for 5 minutes using your own designed alpha for the NYSE dataset with 1402 stocks:
```
timeout 5m bash ./run.sh -a MY_ALPHA -p OUTPUT_PATH  -m initial_alphas/my_alpha_101.txt -s 100000000000 -b NYSE -f 1402
```

## Alpha search commands:

Run an automatic alpha search using previous rounds' searched alphas' outputs on validation dataset as cutoff and using good alphas as initial alphas:
```
./search_alpha_experiments_using_best_alphas.sh -r 2 -o OUTPUT_PATH -b 2 -t 5 -m NYSE -n 1402
```
