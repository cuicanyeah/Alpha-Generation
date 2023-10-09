from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import sklearn
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import task_pb2
import sys
from datetime import datetime
import json
import numpy as np
import pandas as pd
import os
import statistics

class EOD_Preprocessor:
    def __init__(self, data_path, market_name):
        self.data_path = data_path
        self.date_format = '%Y-%m-%d'
        self.market_name = market_name

    def _get_market_dirs(self, ticker_path):
        all_dirs = os.listdir(ticker_path)
        self.market_ticker_dirs = []

        for dir_name in all_dirs:
            if self.market_name == 'NYSE' and ('NYSE' in dir_name or 'price_data_nyse' in dir_name):
                self.market_ticker_dirs.append(dir_name)
            elif self.market_name == 'NASDAQ' and ('NASDAQ' in dir_name or 'price_data_nasdaq' in dir_name):
                self.market_ticker_dirs.append(dir_name)
            elif self.market_name == 'ALL' and (('NASDAQ' in dir_name or 'price_data_nasdaq' in dir_name) or ('NYSE' in dir_name or 'price_data_nyse' in dir_name)):
                self.market_ticker_dirs.append(dir_name)

        return self.market_ticker_dirs
      

    def _read_EOD_data(self, data_path):
        self.data_EOD = []       
        for ticker in self.tickers:
            for market_ticker_dir in self.market_ticker_dirs:
                try:
                    single_EOD = np.genfromtxt(
                        os.path.join(data_path, market_ticker_dir, ticker), dtype=str, delimiter=',',
                        skip_header=True
                    )
                except:
                    continue
                # delete the 'count' feature
                single_EOD = np.delete(single_EOD, [5], axis=1)
                # for a given feature sequence Date,HIGH,CLOSE,LOW,OPEN,VOLUME, rearrange them to Date,HIGH,LOW,OPEN,CLOSE,VOLUME
                permutation = [0, 1, 3, 4, 2, 5]
                single_EOD[:] = single_EOD[:, permutation]
                self.data_EOD.append(single_EOD)

        print('#stocks\' EOD data readin:', len(self.data_EOD))
        assert len(self.tickers) == len(self.data_EOD), 'length of tickers ' \
                                                        'and stocks not match'

    def _read_tickers(self, ticker_fname):
        self.tickers = np.genfromtxt(ticker_fname, dtype=str, delimiter='\t',
                                     skip_header=True)[:, 0]

    def _transfer_EOD_str(self, selected_EOD_str, tra_date_index):
        selected_EOD = np.zeros(selected_EOD_str.shape, dtype=float)
        missing_value_count = 0
        for row, daily_EOD in enumerate(selected_EOD_str):
            missing_value_flag = False
            date_str = daily_EOD[0]
            if date_str in tra_date_index.keys():
                selected_EOD[row][0] = tra_date_index[date_str]
                
                for col in range(1, selected_EOD_str.shape[1]):
                    if daily_EOD[col]: # perform this check because some row could be missing values e,g., daily_EOD col ['2018-05-10' '' '' '' '' '205']
                        selected_EOD[row][col] = float(daily_EOD[col])
                    else:
                        selected_EOD[row][col] = -1234
                        missing_value_flag = True
                if missing_value_flag:
                    missing_value_count += 1
        # print('missing values count across rows and columns: ', missing_value_count)                                                   
        return selected_EOD

    def _get_dates(self, ticker_path, start_date, end_date, topN=3000):
        '''
        From the folder all stocks (NYSE~3000 NASDAQ~5000) select the stocks considered in our work and get a list of dates.
        The logic is to outer merge all stocks of selected time periods into a big df with dates as index, and then delete 
        days where over 80% stocks are missing.
        '''
        # self.market_ticker_dirs is a list because we consider selecting all we have both stock markets (i.e., both dirs)
        self._get_market_dirs(ticker_path)
        init_df = pd.DataFrame()
        count = 0
        avg_volumes = {}
        all_tickers = []

        for market_ticker_dir in self.market_ticker_dirs:
            all_tickers = [file for file in os.listdir(os.path.join(ticker_path, market_ticker_dir)) if file.endswith('.csv')]
            for ticker in all_tickers:
                ticker_df = pd.read_csv(os.path.join(ticker_path, market_ticker_dir, ticker), index_col = 'Date')
                # drop count since count is mainly used for OTC market instruments
                ticker_df = ticker_df.drop(['COUNT'], axis=1)
                
                # Calculate the average volume for the last 3 months before the start date
                three_months_before_start = pd.Timestamp(start_date) - pd.DateOffset(months=3)
                formatted_three_months_before_start = three_months_before_start.strftime('%Y-%m-%d')
                avg_volume = ticker_df.loc[formatted_three_months_before_start:start_date, 'VOLUME'].mean()
                if not pd.isna(avg_volume):  # Only add tickers with valid average volume
                    avg_volumes[ticker] = avg_volume

        # Sort the dictionary by volume in descending order
        sorted_avg_volumes = dict(sorted(avg_volumes.items(), key=lambda item: item[1], reverse=True))

        # Get the top N tickers based on average volume
        top_n_tickers = list(sorted_avg_volumes.keys())[:topN]

        for ticker in all_tickers:
            if ticker not in top_n_tickers:
                continue

            ticker_df = pd.read_csv(os.path.join(ticker_path, market_ticker_dir, ticker), index_col = 'Date')

            if not init_df.empty:
                # adding a suffix to each column name in the ticker_df DataFrame
                ticker_df = ticker_df.add_suffix('__' + ticker[:-4])
                ticker_df = ticker_df[start_date:end_date]

                # use outer join to keep both indices from the original init_df and the new ticker's df
                init_df = init_df.merge(ticker_df, left_index=True, right_index=True, how='outer')
            else:
                init_df = ticker_df
                init_df = init_df.add_suffix('__' + ticker[:-4])
            count += 1

        # check useless indices/dates of a all stocks' df, i.e., init_df, by identifying na rato > 0.8
        all_useful_dates = []
        all_useless_idx = []
        for idx, _ in init_df.iterrows():
            if init_df.loc[[idx]].isna().sum().sum() / len(init_df.columns) > 0.8:
                all_useless_idx.append(idx)
            else:
                all_useful_dates.append(idx)

        init_df = init_df.drop(index=all_useless_idx)

        # # filter stocks
        # all_useful_tickers = []
        # for column in init_df.columns:
        #     if init_df[column].isna().sum() / len(init_df) < 0.02:
        #         all_useful_tickers.append(column.split('__')[-1])

        # all_useful_tickers = list(set(all_useful_tickers))
        # print(all_useful_tickers)

        return all_useful_dates, top_n_tickers

    def check_abnormal_return(self, features, index):
        for j, value in enumerate(features[:, -1]):
            if np.abs(value) > 2 and np.abs(value) != 1234:
                print(f"Abnormal value: {value} at index {j}")
                print(f"Previous row: {features[j-1]}")
                print(f"Current row: {features[j]}")
                if j < len(features) - 1:  # Check to avoid IndexError
                    print(f"Next row: {features[j+1]}")
                else:
                    print("No next row (this is the last row)")
                print(self.tickers[index])
                sys.exit()

    def load_industry_relation_data(self, relation_data_path):
        industry_ticker_file = os.path.join(relation_data_path,
                                            self.market_name + '_industry_ticker.json')
        ticker_index = {}
        for index, ticker in enumerate(self.tickers):
            ticker_index[ticker] = index
        with open(industry_ticker_file, 'r') as fin:
            industry_tickers = json.load(fin)
        print('#industries: ', len(industry_tickers))

        # concatenate lists of all tickers of all industry
        all_tickers_from_industry = sum(industry_tickers.values(), [])
        assert (set(self.tickers) == set(sum(industry_tickers.values(), [])))

        # create indices for the industries
        valid_industry_count = 0
        valid_industry_index = {}
        for industry in industry_tickers.keys():
            valid_industry_index[industry] = valid_industry_count
            valid_industry_count += 1

        # assign the industry index to corresponding stocks of each industry
        industry_all_tickers = {}
        for industry in valid_industry_index.keys():
            cur_ind_tickers = industry_tickers[industry]
            ind_ind = valid_industry_index[industry]
            for i in range(len(cur_ind_tickers)):
                industry_all_tickers[cur_ind_tickers[i]] = ind_ind
        assert (set(self.tickers) == set(list(industry_all_tickers.keys())))        
        return industry_all_tickers

    def generate_feature(self, relation_data_path, if_date_list, selected_tickers_fname, begin_date, end_date,
                         return_days=1, pad_begin=29):
        '''
            Transform the original EOD data collected from Google Finance to a
            friendly format to fit machine learning model via the following steps:
                Calculate moving average (5-days, 10-days, 20-days, 30-days), etc.
                ignoring suspension days (market open, only suspend this stock)
                Normalize features by (feature - min) / (max - min)
        '''        
        self._get_market_dirs(os.path.join(self.data_path))

        if if_date_list:            
            # load from a list of dates to exclude holidays
            trading_dates = np.genfromtxt(
                os.path.join(self.data_path, '..',
                             self.market_name + '_aver_line_dates.csv'),
                dtype=str, delimiter=',', skip_header=False
            )
        else:
            print('getting valid dates and stocks...')
            trading_dates, self.tickers = self._get_dates(os.path.join(self.data_path), begin_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        print('#trading dates:', len(trading_dates))
        print('begin date:', begin_date)

        # make a dictionary to translate dates into index and vice versa
        tra_dates_index = {}
        for index, date in enumerate(trading_dates):
            tra_dates_index[date] = index

        if selected_tickers_fname:
            self.tickers = np.genfromtxt(
                os.path.join(self.data_path, '..', selected_tickers_fname),
                dtype=str, delimiter='\t', skip_header=False
            )

        print('#tickers selected:', len(self.tickers))
        self._read_EOD_data(self.data_path)
        stocks_features = []
        labels = []

        ## load stock industry relation data (TO DO)
        # industry_all_tickers = self.load_industry_relation_data(relation_data_path)

        for index, ticker in enumerate(self.tickers):
            print('ticker: ', ticker)
            # industry_relation = industry_all_tickers[ticker]

            # The indices of self.tickers and self.data_EOD match.
            # So we use the same index and then fetch EOD data.
            # stock_index = ticker_index[ticker]
            single_EOD = self.data_EOD[index]

            begin_date_row = -1
            for date_index, daily_EOD in enumerate(single_EOD):
                date_str = daily_EOD[0]
                cur_date = datetime.strptime(date_str, self.date_format)

                # from original raw stock price data file extract data,
                # if larger than our specified begin date then extract.
                if cur_date > begin_date:
                    begin_date_row = date_index
                    break
            for date_index, daily_EOD in enumerate(single_EOD):
                date_str = daily_EOD[0]
                cur_date = datetime.strptime(date_str, self.date_format)

                # from original raw stock price data file extract data,
                # if larger than our specified begin date then extract.
                if cur_date > end_date:
                    end_date_row = date_index
                    break

            selected_EOD_str = single_EOD[begin_date_row:end_date_row]

            # transfer the indices of date string into indices of numbers
            selected_EOD = self._transfer_EOD_str(selected_EOD_str,
                                                  tra_dates_index)

            # calculate moving average features
            begin_date_row = -1
            for row in selected_EOD[:, 0]:
                row = int(row)

                # offset for the first 30-days average
                if row >= pad_begin:
                    begin_date_row = row
                    break
            mov_aver_features = np.zeros(
                [selected_EOD.shape[0], 8], dtype=float
            )

            # 5-, 10-, 20-, 30-days average and volatility
            for row in range(begin_date_row, selected_EOD.shape[0]):
                date_index = selected_EOD[row][0]
                aver_5 = 0.0
                aver_10 = 0.0
                aver_20 = 0.0
                aver_30 = 0.0
                count_5 = 0
                count_10 = 0
                count_20 = 0
                count_30 = 0
                std_5 = []
                std_10 = []
                std_20 = []
                std_30 = []
                for offset in range(30):
                    date_gap = date_index - selected_EOD[row - offset][0]
                    if date_gap < 5:
                        count_5 += 1
                        aver_5 += selected_EOD[row - offset][4]
                        std_5.append(selected_EOD[row - offset][4])
                    if date_gap < 10:
                        count_10 += 1
                        aver_10 += selected_EOD[row - offset][4]
                        std_10.append(selected_EOD[row - offset][4])
                    if date_gap < 20:
                        count_20 += 1
                        aver_20 += selected_EOD[row - offset][4]
                        std_20.append(selected_EOD[row - offset][4])
                    if date_gap < 30:
                        count_30 += 1
                        aver_30 += selected_EOD[row - offset][4]
                        std_30.append(selected_EOD[row - offset][4])

                # some data such as ticker DWAQ in period of 2016-12-06~21 missing data 15 days
                averages_data = [(aver_5, count_5, 0), (aver_10, count_10, 1), (aver_20, count_20, 2), (aver_30, count_30, 3)]
                stds_data = [(std_5, 4), (std_10, 5), (std_20, 6), (std_30, 7)]

                for average, count, index in averages_data:
                    mov_aver_features[row][index] = average / count if count != 0 else -1234

                for std_values, index in stds_data:
                    mov_aver_features[row][index] = statistics.stdev(std_values) if len(std_values) > 1 else -1234

            # check extreme price values
            pri_min = np.min(selected_EOD[begin_date_row:, 4])
            price_max = np.max(selected_EOD[begin_date_row:, 4])

            print(self.tickers[index], 'minimum:', pri_min,
                  'maximum:', price_max, 'ratio:', price_max / pri_min)
            if price_max / pri_min > 10:
                print('!!!!!!!!!')

            '''
                generate feature and ground truth in the following format:
                date_index, 5-day, 10-day, 20-day, 30-day, close price
                two ways to pad missing dates:
                for dates without record, pad a row [date_index, -1234 * 5]
            '''
            features = np.ones([len(trading_dates) - pad_begin, 15],
                               dtype=float) * -1234
            rows = np.ones([len(trading_dates) - pad_begin, 1],
                           dtype=float)
            # data missed at the beginning
            for row in range(len(trading_dates) - pad_begin):
                rows[row][0] = row
            for row in range(begin_date_row, selected_EOD.shape[0]):
                cur_index = int(selected_EOD[row][0])
                features[cur_index - pad_begin][0:8] = mov_aver_features[
                    row]

                '''adding the next if condition because of the above mentioned prob - index might not be continuous. 
                    Only if continuous will add features'''
                if cur_index - int(selected_EOD[row - return_days][0]) == \
                        return_days:
                    features[cur_index - pad_begin][-7:-2] = \
                        selected_EOD[row][1:]
                    if (row + return_days) < selected_EOD.shape[0]:
                        price_diff = selected_EOD[row + return_days][-2] - selected_EOD[row][-2]
                        if selected_EOD[row][-2] == 0: # selected_EOD[row + return_days][-2] == 0 or comment off because unlike log return raw return can be 0
                            features[cur_index - pad_begin][-1] = -1234
                        else:
                            '''
                            Using -1234 as missing value's representation needs to take care of extreme values. In the following example of debugging information, 
                            one missing value cause two returns to be extreme. So must assign -1234 to the corresponding returns. E.g.,
                            "
                            selected_EOD[row + return_days][-2] -1234.0
                            selected_EOD[row][-2] 40.83
                            Abnormal return detected! Calculated price diff: -1274.83 too high! Abnormal return -31.222875336762183
                            selected_EOD[row + return_days][-2] 37.74
                            selected_EOD[row][-2] -1234.0
                            Abnormal return detected! Calculated price diff: 1271.74 too high! Abnormal return -1.030583468395462
                            "
                            '''
                            if np.abs(price_diff/selected_EOD[row][-2]) > 1:
                                print('today close price', selected_EOD[row + return_days][-2])
                                print('yesterday close price', selected_EOD[row][-2])
                                print('Abnormal return detected! Calculated price diff: {} too high! Abnormal return {}'.format(price_diff, price_diff/selected_EOD[row][-2]))
                                features[cur_index - pad_begin][-1] = -1234
                            else:
                                features[cur_index - pad_begin][-1] = price_diff / selected_EOD[row][-2]
                        
                        # check if there is any abnormal return
                        # assert np.abs(price_diff/selected_EOD[row][-2]) <= 1, f'Abnormal return detected! Calculated price diff: {price_diff:.2f} too high! Abnormal return {price_diff/selected_EOD[row][-2]:.2f}'

            # for some stocks (e.g., the stock DWPP) whose 0 is missing values. 
            # so except for the label return column, replace with dummy value of -1234 to mark it as missing.
            if (features[:, :-1] == 0).any():
                features[:, :-1][features[:, :-1] == 0] = -1234
                print('all missing values number: ', np.sum(list(features == -1234)))

            # -2 is the industry index column
            # features[:, -2] = industry_relation
            features[:, -2] = 0

            # last row all missing values
            features = np.delete(features, -1, 0)
            # because the last row don't have label since no tomorrow data
            features = np.delete(features, -1, 0)

            labels.append(features[:, -1:])

            # Check for abnormal returns
            self.check_abnormal_return(features, index)

            features = np.delete(features, 14, 1)

            # check missing values are handled correctly
            assert (sum(list(features[:, 13] == -1234)) == 0)
            assert (sum(list(features[:, -1] == -1234)) == 0)

            stocks_features.append(features)

        # check abnormal return values
        for i in range(len(labels)):
            for j in range(len(labels[i])):
                if np.abs(labels[i][j]) > 1 and np.abs(labels[i][j]) != 1234:
                    print('> 1!!!!!!!!!!', labels[i][j])
                    print('index', j)
                    print(self.tickers[i])
                    # sys.exit()
        
        np.save(self.market_name+'_stocks_features.npy', stocks_features)
        np.save(self.market_name+'_labels.npy', labels)
        return stocks_features, labels


def create_dataset(
    dataset_name, stock_tasks,
    num_train_examples, num_valid_examples, num_test_examples,
    seed, dataset_dict):
    """Create a dataset from the given spec."""
    data, labels = get_dataset(
        dataset_name,
        dataset_dict,
        stock_tasks)

    (train_data, train_labels, valid_data, valid_labels,
     test_data, test_labels) = train_valid_test_split(
         data, labels,
         num_train_examples,
         num_valid_examples,
         num_test_examples,
         seed)

    dataset = task_pb2.ScalarLabelDataset()
    for i in range(train_data.shape[0]):
        train_feature = dataset.train_features.add()
        train_feature.features.extend(list(train_data[i]))
        dataset.train_labels.append(train_labels[i][0])
        # check if there is any abnormal return
        assert not (np.abs(train_labels[i]) > 2 and np.abs(train_labels[i]) != 1234), f'Abnormal return detected! Calculated price diff: {train_labels[i]:.2f} too high!'

    for i in range(valid_data.shape[0]):
        valid_feature = dataset.valid_features.add()
        valid_feature.features.extend(list(valid_data[i]))
        dataset.valid_labels.append(valid_labels[i][0])
        # check if there is any abnormal return
        assert not (np.abs(valid_labels[i]) > 2 and np.abs(valid_labels[i]) != 1234), f'Abnormal return detected! Calculated price diff: {valid_labels[i]:.2f} too high!'

    if test_data is not None:
        for i in range(test_data.shape[0]):
            test_feature = dataset.test_features.add()
            test_feature.features.extend(list(test_data[i]))
            dataset.test_labels.append(test_labels[i][0])
    return dataset


def load_dataset(saved_dataset):
    """Load the dataset saved in a ScalarLabelDataset proto."""
    num_train = len(saved_dataset.train_labels)
    assert len(saved_dataset.train_labels) == len(saved_dataset.train_features)
    num_valid = len(saved_dataset.valid_labels)
    assert len(saved_dataset.valid_labels) == len(saved_dataset.valid_features)
    num_test = len(saved_dataset.test_labels)
    assert len(saved_dataset.test_labels) == len(saved_dataset.test_features)
    if num_train == 0 or num_valid == 0:
        raise ValueError('Number of train/valid examples'
                         ' must be more than zero.')
    feature_size = len(saved_dataset.train_features[0].features)

    train_data = np.zeros((num_train, feature_size))
    train_labels = np.zeros(num_train)
    for i in range(num_train):
        train_labels[i] = saved_dataset.train_labels[i]
        for j in range(feature_size):
            train_data[i][j] = saved_dataset.train_features[i].features[j]

    valid_data = np.zeros((num_valid, feature_size))
    valid_labels = np.zeros(num_valid)
    for i in range(num_valid):
        valid_labels[i] = saved_dataset.valid_labels[i]
        for j in range(feature_size):
            valid_data[i][j] = saved_dataset.valid_features[i].features[j]

    if num_test > 0:
        test_data = np.zeros((num_test, feature_size))
        test_labels = np.zeros(num_test)
        for i in range(num_test):
            test_labels[i] = saved_dataset.test_labels[i]
            for j in range(feature_size):
                test_data[i][j] = saved_dataset.test_features[i].features[j]
    else:
        test_data = None
        test_labels = None

    return (train_data, train_labels, valid_data, valid_labels,
            test_data, test_labels)


def get_dataset(
    name, dataset_dict, stock_tasks=None):

    train_data = dataset_dict[name]['features']
    train_labels = dataset_dict[name]['labels']

    # train_data = train_data.astype(np.float)
    assert len(train_data) == len(train_labels)

    if stock_tasks is not None:
        train_data = train_data[stock_tasks]
        train_labels = train_labels[stock_tasks]

    assert len(train_data) == len(train_labels)

    return train_data, train_labels

def train_valid_test_split(
    data, labels,
    num_train_examples, num_valid_examples, num_test_examples,
        seed, use_stratify=False):
    """Split data into train, valid and test with given seed."""
    # num_test_examples is set to 0 in our case
    if num_test_examples > 0:
        train_data, test_data, train_labels, test_labels = (
            sklearn.model_selection.train_test_split(
                data, labels,
                train_size=(
                    num_train_examples +
                    num_valid_examples),
                test_size=num_test_examples,
                random_state=seed, shuffle=False))

    else:
        train_data, train_labels = data, labels
        test_data = None
        test_labels = None

    # the valid and test datasets are put together and later seperated in cpp codes
    train_data, valid_data, train_labels, valid_labels = (
        sklearn.model_selection.train_test_split(
            train_data, train_labels,
            train_size=num_train_examples,
            test_size=num_valid_examples,
            random_state=seed, shuffle=False))
    return (
        train_data, train_labels,
        valid_data, valid_labels,
        test_data, test_labels)


flags.DEFINE_string(
    'input_path', 'data/eikon_data',
    'Path of EOD data.')

flags.DEFINE_string(
    'output_path', 'data_for_a_stock_market',
    'Path of the folder to save the datasets.')

flags.DEFINE_string(
    'relation_data_path', 'data/relation/sector_industry',
    'Path of EOD data.')

flags.DEFINE_string(
    'market', 'ALL',
    'Market name.')

flags.DEFINE_integer('num_train_examples', 1000,
                     'Number of training examples in each dataset.')

flags.DEFINE_integer('num_valid_examples', 228,
                     'Number of validation examples in each dataset.')

# set to 0 here because we split num_valid_examples 244 into valid and test later in cpp
flags.DEFINE_integer('num_test_examples', 0,
                     'Number of test examples in each dataset.')

flags.DEFINE_integer('projected_dim', 13,
                     'The dimensionality to project the data into.')

flags.DEFINE_string('dataset_name', 'ALL',
                    'Name of the dataset.')

flags.DEFINE_string('start_date', '2017-1-2',
                    'start_date.')

flags.DEFINE_string('end_date', '2021-12-31',
                    'end_date.')

flags.DEFINE_integer('min_data_seed', 0,
                     'Generate one dataset for each seed in '
                     '[min_data_seed, max_data_seed).')

flags.DEFINE_integer('max_data_seed', 1,
                     'Generate one dataset for each seed in '
                     '[min_data_seed, max_data_seed).')

flags.DEFINE_integer('if_date_list', 0,
                     '0 means not provided and we need to preprocess '
                     '1 means it is provided.')

flags.DEFINE_integer('if_ticker_list', 0,
                     '0 means not provided and we need to preprocess '
                     '1 means it is provided.')

FLAGS = flags.FLAGS


def main(argv):
    """Create and save the datasets."""
    processor = EOD_Preprocessor(FLAGS.input_path, FLAGS.market)

    ticker_list = ''
    if FLAGS.if_ticker_list:
        ticker_list = processor.market_name + '_tickers_qualify_dr-0.98_min-5_smooth.csv'

    len_tickers = 0
    if os.path.isfile(processor.market_name+'_stocks_features.npy') and os.path.isfile(processor.market_name+'_labels.npy'):
        stock_features = np.load(processor.market_name+'_stocks_features.npy')
        stock_labels = np.load(processor.market_name+'_labels.npy')
        print('# of stocks: ', len(stock_features))
        print('stock_features.npy shape: ', np.shape(stock_features[0]))
        FLAGS.num_valid_examples = np.shape(stock_features[0])[0] - FLAGS.num_train_examples
        len_tickers = len(stock_features)     
    else:
        stock_features, stock_labels = processor.generate_feature(
            FLAGS.relation_data_path,
            FLAGS.if_date_list,
            ticker_list,
            datetime.strptime(FLAGS.start_date, processor.date_format),
            datetime.strptime(FLAGS.end_date, processor.date_format),
            return_days=1,
            pad_begin=29
        )
        len_tickers = len(processor.tickers)

    stock_tasks = list(range(0, len_tickers))

    if not os.path.exists(FLAGS.output_path):
        os.makedirs(FLAGS.output_path)

    name = processor.market_name
    dataset_dict = {'features': stock_features, 'labels': stock_labels}
    dataset_dict[name] = dataset_dict

    for stock_task in stock_tasks:
        random_seeds = range(FLAGS.min_data_seed, FLAGS.max_data_seed)
        for seed in random_seeds:
            dataset = create_dataset(
                processor.market_name, stock_task,
                FLAGS.num_train_examples, FLAGS.num_valid_examples,
                FLAGS.num_test_examples, seed, dataset_dict)
            filename = 'dataset_{}-stock_{}-dim_{}-seed_{}'.format(
                processor.market_name, stock_task,
                FLAGS.projected_dim, seed)
            serialized_dataset = dataset.SerializeToString()

            with open(os.path.join(FLAGS.output_path, filename), 'wb') as f:
                f.write(serialized_dataset)


if __name__ == '__main__':
    app.run(main)
