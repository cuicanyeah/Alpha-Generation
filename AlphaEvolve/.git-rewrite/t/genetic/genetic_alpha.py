import numpy as np
import pandas as pd
from scipy.stats import rankdata
import pickle
from gplearn import genetic
from gplearn.functions import make_function
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor
from gplearn.fitness import make_fitness
from evaluator_long_short_portfolio import evaluate_sr
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
import time
from absl import flags
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import re
import matplotlib.pyplot as plt
import sys
from absl import app

flags.DEFINE_string('data_path', "",
                     'Path to get data.')

flags.DEFINE_string('output_path', "",
                  'Path to save output.')

flags.DEFINE_integer('ith_round', 1,
                  'ith round.')

flags.DEFINE_string('save_load', "s",
                  'save or load.')

flags.DEFINE_string('cutoffs_valid', "",
                  'save or load.')

flags.DEFINE_string('cutoffs_test', "",
                  'save or load.')

FLAGS = flags.FLAGS

def main(argv):
    start = time.time()
    print("time start...")

    fields = ['mv5', 'mv10', 'mv20', 'mv30', 'vol5', 'vol10', 'vol20', 'vol30', 'open', 'high', 'low', 'close', 'volume']

    """
    Available individual functions are:
    ‘add’ : addition, arity=2.
    ‘sub’ : subtraction, arity=2.
    ‘mul’ : multiplication, arity=2.
    ‘div’ : protected division where a denominator near-zero returns 1., arity=2.
    ‘sqrt’ : protected square root where the absolute value of the argument is used, arity=1.
    ‘log’ : protected log where the absolute value of the argument is used and a near-zero argument returns 0., arity=1.
    ‘abs’ : absolute value, arity=1.
    ‘neg’ : negative, arity=1.
    ‘inv’ : protected inverse where a near-zero argument returns 0., arity=1. 
    ‘max’ : maximum, arity=2.
    ‘min’ : minimum, arity=2.
    ‘sin’ : sine (radians), arity=1.
    ‘cos’ : cosine (radians), arity=1.
    ‘tan’ : tangent (radians), arity=1.
    """

    init_function = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min', 'sin', 'cos', 'tan']
    # init_function = ['add', 'sub', 'mul', 'div']

    num_stock = 1026
    test_num_steps = 116
    train_num_steps = 1244 - test_num_steps


    def _rolling_rank(data):
        value = rankdata(data)[-1]

        return value


    def _rolling_prod(data):
        return np.prod(data)


    def _ts_sum(data):
        window = 10
        # larger than 10000 is to pass a sanity check by ga algorithm
        # if data.shape[0] > 10000000000000000000:
        if data.shape[0] > 10000:
            if data.shape[0] > 1020000:
                num_steps = train_num_steps
            else:
                num_steps = test_num_steps
            all_stocks_data = []
            for i in range(num_stock):
                ith_data = data[i * num_steps: (i+1) * num_steps]
                ith_data_value = np.array(pd.Series(ith_data.flatten()).rolling(window).sum().tolist())
                ith_data_value = np.nan_to_num(ith_data_value)
                all_stocks_data.append(ith_data_value)
            value = np.hstack(all_stocks_data)
            assert value.shape[0] == data.shape[0]
            return value
        else:
            return data


    def _sma(data):
        window = 10
        # if data.shape[0] > 1000000000000000000:
        if data.shape[0] > 10000:
            if data.shape[0] > 1020000:
                num_steps = train_num_steps
            else:
                num_steps = test_num_steps
            all_stocks_data = []
            for i in range(num_stock):
                ith_data = data[i * num_steps: (i+1) * num_steps]
                ith_data_value = np.array(pd.Series(ith_data.flatten()).rolling(window).mean().tolist())
                ith_data_value = np.nan_to_num(ith_data_value)
                all_stocks_data.append(ith_data_value)
            value = np.hstack(all_stocks_data)
            assert value.shape[0] == data.shape[0]
            return value
        else:
            return data

    def _stddev(data):
        window = 10
        if data.shape[0] > 10000:
            if data.shape[0] > 1020000:
                num_steps = train_num_steps
            else:
                num_steps = test_num_steps
            all_stocks_data = []
            for i in range(num_stock):
                ith_data = data[i * num_steps: (i+1) * num_steps]
                ith_data_value = np.array(pd.Series(ith_data.flatten()).rolling(window).std().tolist())
                ith_data_value = np.nan_to_num(ith_data_value)
                all_stocks_data.append(ith_data_value)
            value = np.hstack(all_stocks_data)
            assert value.shape[0] == data.shape[0]
            return value
        else:
            return data

    def _ts_rank(data):
        window = 10
        # if data.shape[0] > 1000000000000000000:
        if data.shape[0] > 10000:
            if data.shape[0] > 1020000:
                num_steps = train_num_steps
            else:
                num_steps = test_num_steps
            all_stocks_data = []
            for i in range(num_stock):
                ith_data = data[i * num_steps: (i+1) * num_steps]
                ith_data_value = np.array(pd.Series(ith_data.flatten()).rolling(window).apply(_rolling_rank).tolist())
                ith_data_value = np.nan_to_num(ith_data_value)
                all_stocks_data.append(ith_data_value)
            value = np.hstack(all_stocks_data)
            assert value.shape[0] == data.shape[0]
            return value
        else:
            return data

    def _product(data):
        window = 10
        # if data.shape[0] > 10000000000000000000:
        if data.shape[0] > 10000:
            if data.shape[0] > 1020000:
                num_steps = train_num_steps
            else:
                num_steps = test_num_steps
            all_stocks_data = []
            for i in range(num_stock):
                ith_data = data[i * num_steps: (i+1) * num_steps]
                ith_data_value = np.array(pd.Series(ith_data.flatten()).rolling(window).apply(_rolling_prod).tolist())
                ith_data_value = np.nan_to_num(ith_data_value)
                all_stocks_data.append(ith_data_value)
            value = np.hstack(all_stocks_data)
            assert value.shape[0] == data.shape[0]
            return value
        else:
            return data

    def _ts_min(data):
        window = 10
        # if data.shape[0] > 100000000000000000000:
        if data.shape[0] > 10000:
            if data.shape[0] > 1020000:
                num_steps = train_num_steps
            else:
                num_steps = test_num_steps
            all_stocks_data = []
            for i in range(num_stock):
                ith_data = data[i * num_steps: (i+1) * num_steps]
                ith_data_value = np.array(pd.Series(ith_data.flatten()).rolling(window).min().tolist())
                ith_data_value = np.nan_to_num(ith_data_value)
                all_stocks_data.append(ith_data_value)
            value = np.hstack(all_stocks_data)
            assert value.shape[0] == data.shape[0]
            return value
        else:
            return data

    def _ts_max(data):
        window = 10
        # if data.shape[0] > 100000000000000000:
        if data.shape[0] > 10000:
            if data.shape[0] > 1020000:
                num_steps = train_num_steps
            else:
                num_steps = test_num_steps
            all_stocks_data = []
            for i in range(num_stock):
                ith_data = data[i * num_steps: (i+1) * num_steps]
                ith_data_value = np.array(pd.Series(ith_data.flatten()).rolling(window).max().tolist())
                ith_data_value = np.nan_to_num(ith_data_value)
                all_stocks_data.append(ith_data_value)
            value = np.hstack(all_stocks_data)
            assert value.shape[0] == data.shape[0]
            return value
        else:
            return data

    def _delta(data):
        window = 10
        # if data.shape[0] > 10000000000000000:
        if data.shape[0] > 10000:
            if data.shape[0] > 1020000:
                num_steps = train_num_steps
            else:
                num_steps = test_num_steps

            all_stocks_data = []
            for i in range(num_stock):
                ith_data = data[i * num_steps: (i+1) * num_steps]
                ith_data_value = np.diff(ith_data.flatten())
                ith_data_value = np.append(0, ith_data_value)
                all_stocks_data.append(ith_data_value)
            value = np.hstack(all_stocks_data)
            assert value.shape[0] == data.shape[0]
            return value
        else:
            return data

    def _delay(data):
        period = 1
        # if data.shape[0] > 1000000000000000:
        if data.shape[0] > 10000:
            if data.shape[0] > 1020000:
                num_steps = train_num_steps
            else:
                num_steps = test_num_steps
            all_stocks_data = []
            for i in range(num_stock):
                ith_data = data[i * num_steps: (i+1) * num_steps]
                ith_data_value = np.array(pd.Series(ith_data.flatten()).shift(1).tolist())
                ith_data_value = np.nan_to_num(ith_data_value)
                all_stocks_data.append(ith_data_value)
            value = np.hstack(all_stocks_data)
            assert value.shape[0] == data.shape[0]
            return value
        else:
            return data


    def _scale(data):
        k = 1
        # if data.shape[0] > 10000000000000000000:
        if data.shape[0] > 10000:
            if data.shape[0] > 1020000:
                num_steps = train_num_steps
            else:
                num_steps = test_num_steps
            all_stocks_data = []
            for i in range(num_stock):
                ith_data = data[i * num_steps: (i+1) * num_steps]
                ith_data = pd.Series(ith_data.flatten())
                ith_data_value = ith_data.mul(1).div(np.abs(ith_data).sum())
                ith_data_value = np.nan_to_num(ith_data_value)
                all_stocks_data.append(ith_data_value)
            value = np.hstack(all_stocks_data)
            assert value.shape[0] == data.shape[0]
            return value
        else:
            return data

    def _ts_argmax(data):
        window = 10
        value = pd.Series(data.flatten()).rolling(10).apply(np.argmax) + 1
        value = np.nan_to_num(value)

        return value


    def _ts_argmin(data):
        window = 10
        value = pd.Series(data.flatten()).rolling(10).apply(np.argmin) + 1
        value = np.nan_to_num(value)

        return value


    def _all_rank(data):
        # if data.shape[0] > 100000000000000000000:
        if data.shape[0] > 10000:
            if data.shape[0] > 1020000:
                num_steps = train_num_steps
            else:
                num_steps = test_num_steps
            all_stocks_data = np.full(data.shape, -1)
            for i in range(num_steps):
                stocks_ind = np.array([j for j, n in enumerate(list(data)) if j % num_steps == i])
                ith_data = data[stocks_ind]
                temp = ith_data.argsort()
                ranks = np.empty_like(temp)
                ranks[temp] = np.arange(len(ith_data))
                all_stocks_data[stocks_ind] = ranks
            assert np.min(all_stocks_data) > -1
            assert all_stocks_data.shape[0] == data.shape[0]
            return all_stocks_data + 1 # because this way of ranking gives lowest 0, add to make it same as other ops nonzero
        else:
            return data

    def _my_metric(y, y_pred, w):
        if y.shape[0] > 1000:
            value, performance = evaluate_sr(y_pred, y, w, False, FLAGS.cutoffs_valid, FLAGS.cutoffs_test, FLAGS.ith_round)
            value_ret = value['correlation']
        else:
            value_ret = 1
        return value_ret


    stocks_features = np.load('stocks_features.npy')
    labels = np.load('labels.npy')

    stocks_features_train = []
    stocks_features_test = []
    labels_train = []
    labels_test = []

    for i in range(stocks_features.shape[0]):
        stocks_features_train.append(stocks_features[i][:train_num_steps])
        stocks_features_test.append(stocks_features[i][-test_num_steps:])
        labels_train.append(labels[i][:train_num_steps])
        labels_test.append(labels[i][-test_num_steps:])

    stocks_features_train = np.delete(stocks_features_train, 13, 2)
    labels_train = np.vstack(labels_train)
    stocks_features_train = np.vstack(stocks_features_train)
    stocks_features_test = np.delete(stocks_features_test, 13, 2)
    labels_test = np.vstack(labels_test)
    stocks_features_test = np.vstack(stocks_features_test)

    mask_train = np.full(labels_train.shape, 1)
    for i in range(stocks_features_train.shape[0]):
        for j in range(stocks_features_train.shape[1]):
            if np.abs(stocks_features_train[i][j]) == 1234 or np.abs(labels_train[i]) == 1234:
                mask_train[i] = 0
                break
    mask_test = np.full(labels_test.shape, 1)
    for i in range(stocks_features_test.shape[0]):
        for j in range(stocks_features_test.shape[1]):
            if np.abs(stocks_features_test[i][j]) == 1234 or np.abs(labels_test[i]) == 1234:
                mask_test[i] = 0
                break

    delta = make_function(function=_delta, name='delta', arity=1)
    delay = make_function(function=_delay, name='delay', arity=1)
    scale = make_function(function=_scale, name='scale', arity=1)
    sma = make_function(function=_sma, name='sma', arity=1)
    stddev = make_function(function=_stddev, name='stddev', arity=1)
    product = make_function(function=_product, name='product', arity=1)
    ts_rank = make_function(function=_ts_rank, name='ts_rank', arity=1)
    ts_min = make_function(function=_ts_min, name='ts_min', arity=1)
    ts_max = make_function(function=_ts_max, name='ts_max', arity=1)
    ts_argmax = make_function(function=_ts_argmax, name='ts_argmax', arity=1)
    ts_argmin = make_function(function=_ts_argmin, name='ts_argmin', arity=1)
    ts_sum = make_function(function=_ts_sum, name='ts_sum', arity=1)
    all_rank = make_function(function=_all_rank, name='all_rank', arity=1)

    user_function = [delta, delay, scale, sma, stddev, product, ts_rank, ts_min, ts_max, ts_argmax, ts_argmin, ts_sum,
                     all_rank]

    my_metric = make_fitness(function=_my_metric, greater_is_better=True)

    generations = 1
    function_set = init_function + user_function
    metric = my_metric
    population_size = 100
    random_state = 0
    
    est_gp = SymbolicTransformer(
                                feature_names=fields,
                                function_set=function_set,
                                generations=generations,
                                metric=metric,
                                population_size=population_size,
                                tournament_size=10,
                                n_jobs=1,
                                verbose=1,
                                random_state=random_state,
                             )
	
    est_gp.fit(stocks_features_train, labels_train.ravel(), mask_train)
        
    best_programs = est_gp._best_programs
    best_programs_dict = {}
    for p in best_programs:
        factor_name = 'alpha_' + str(best_programs.index(p) + 1)
        best_programs_dict[factor_name] = {'fitness': p.fitness_, 'expression': str(p), 'depth': p.depth_,
                                           'length': p.length_}
    print('result view 1: ',best_programs_dict)
    best_programs_dict = pd.DataFrame(best_programs_dict).T
    print('result view 2',best_programs_dict)
    best_programs_dict = best_programs_dict.sort_values(by='fitness')

    # print validate result
    final_predict_validate = est_gp.transform(stocks_features_train)

    for i in range(final_predict_validate.shape[1]):
        if i == 0:
            print('validate performance: ')
            k = final_predict_validate[:, i][..., np.newaxis]
            final_performance, strategy_returns = evaluate_sr(k, labels_train, mask_train, False, FLAGS.cutoffs_valid, FLAGS.cutoffs_test, FLAGS.ith_round)
            print(final_performance)
            vperf_path_and_name = FLAGS.output_path + '/genetic_baseline_' + str(FLAGS.ith_round) + 'th_round_validate_performance.pkl'
            vtest_path_and_name = FLAGS.output_path + '/genetic_baseline_' + str(FLAGS.ith_round) + 'th_round_validate_returns.pkl'
            with open(vperf_path_and_name, 'wb') as fp:
                pickle.dump(final_performance, fp, protocol=pickle.HIGHEST_PROTOCOL)
            with open(vtest_path_and_name, 'wb') as fp:
                pickle.dump(strategy_returns, fp, protocol=pickle.HIGHEST_PROTOCOL)
    # print test result
    final_predict = est_gp.transform(stocks_features_test)
    print(final_predict.shape)
    for i in range(final_predict.shape[1]):
        if i == 0:
            print('test performance: ')
            k = final_predict[:, i][..., np.newaxis]
            final_performance, strategy_returns = evaluate_sr(k, labels_test, mask_test, True, FLAGS.cutoffs_valid, FLAGS.cutoffs_test, FLAGS.ith_round)
            print(final_performance)
            tperf_path_and_name = FLAGS.output_path + '/genetic_baseline_' + str(FLAGS.ith_round) + 'th_round_test_performance.pkl'
            ttest_path_and_name = FLAGS.output_path + '/genetic_baseline_' + str(FLAGS.ith_round) + 'th_round_test_returns.pkl'
            with open(tperf_path_and_name, 'wb') as fp:
                pickle.dump(final_performance, fp, protocol=pickle.HIGHEST_PROTOCOL)
            with open(ttest_path_and_name, 'wb') as fp:
                pickle.dump(strategy_returns, fp, protocol=pickle.HIGHEST_PROTOCOL)
    end = time.time()
    print('final time...')
    print(end - start)

    def alpha_factor_graph(num):
        factor = best_programs[num - 1]
        print(factor)
        print('fitness: {0}, depth: {1}, length: {2}'.format(factor.fitness_, factor.depth_, factor.length_))

        dot_data = factor.export_graphviz()
        graph = graphviz.Source(dot_data)
        graph.render('images/alpha_factor_graph', format='png', cleanup=True)

        return graph

if __name__ == '__main__':
    app.run(main)
