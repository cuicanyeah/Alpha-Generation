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
from sklearn.model_selection import train_test_split

class EOD_Preprocessor:
    def __init__(self, data_path, market_name):
        self.data_path = data_path
        self.date_format = '%Y-%m-%d'
        self.market_name = market_name

    def _get_market_dir(self, ticker_path):
        all_dirs = os.listdir(ticker_path)
        self.market_ticker_dir = ''
        for dir_name in all_dirs:
            if self.market_name == 'NYSE' and ('NYSE' in dir_name or 'nyse' in dir_name):
                self.market_ticker_dir = dir_name 
            elif self.market_name == 'NASDAQ' and ('NASDAQ' in dir_name or 'nasdaq' in dir_name):
                self.market_ticker_dir = dir_name        

    def _read_EOD_data(self, data_path):
        self.data_EOD = []
        for index, ticker in enumerate(self.tickers):
            single_EOD = np.genfromtxt(
                os.path.join(data_path, self.market_ticker_dir, ticker + '.csv'), dtype=str, delimiter=',',
                skip_header=True
            )
            # delete the 'count' feature
            single_EOD = np.delete(single_EOD, [5], axis=1)
            
            permutation = [0, 1, 3, 4, 2, 5]
            idx = np.empty_like(permutation)
            idx[permutation] = np.arange(len(permutation))

            single_EOD[:] = single_EOD[:, idx]
            self.data_EOD.append(single_EOD)

        print('#stocks\' EOD data readin:', len(self.data_EOD))
        assert len(self.tickers) == len(self.data_EOD), 'length of tickers ' \
                                                        'and stocks not match'

    def _read_tickers(self, ticker_fname):
        self.tickers = np.genfromtxt(ticker_fname, dtype=str, delimiter='\t',
                                     skip_header=True)[:, 0]

    def _transfer_EOD_str(self, selected_EOD_str, tra_date_index):
        selected_EOD = np.zeros(selected_EOD_str.shape, dtype=float)
        count = 0
        for row, daily_EOD in enumerate(selected_EOD_str):
            flag = False
            date_str = daily_EOD[0]
            if date_str in tra_date_index.keys():
                selected_EOD[row][0] = tra_date_index[date_str]
                
                for col in range(1, selected_EOD_str.shape[1]):
                    if daily_EOD[col]:
                        selected_EOD[row][col] = float(daily_EOD[col])
                    else:
                        selected_EOD[row][col] = -1234
                        flag = True
                if flag:
                    count += 1
                    # else:
                    #     print('daily_EOD col', daily_EOD)
                    #     import sys
                    #     sys.exit() 
        print('count: ', count)                                                   
        return selected_EOD

    def _get_dates(self, ticker_path, start_date, end_date):

        market_ticker_path = os.path.join(ticker_path, self.market_ticker_dir)
        all_tickers = os.listdir(market_ticker_path)        

        '''
        NEW FEATURE: count is mainly used for OTC market instruments - where there is no volume 
        indicated eg FX market and it represents number of quotes. For an exchange 
        traded instrument this is number of trades.
        '''
        
        all_tickers_df = {}
        init_df = pd.DataFrame()
        count = 0
        for ticker in all_tickers:
            ticker_df = pd.read_csv(os.path.join(market_ticker_path, ticker), index_col = 'Date')
            ticker_df = ticker_df.drop(['COUNT'], axis=1)
            all_tickers_df[ticker] = ticker_df
            if not init_df.empty:
                ticker_df = ticker_df.add_suffix('__' + ticker[:-4])

                # print(start_date)
                # start_year = start_date[-4:]
                # print('start_year', start_year)
                # end_year = end_date[-4:]
                # for idx, x in init_df.iterrows():
                #     if start_year in idx:
                #         start_date = idx
                #         break
                # flag = False
                # for idx, x in init_df.iterrows():
                #     if end_year in idx:
                #         end_date = idx
                #         flag = True
                #     if end_year not in idx and flag:
                #         break

                # print('start_date', start_date)
                # print('end_date', end_date)

                ticker_df = ticker_df[start_date:end_date]
                # print('ticker_df ', ticker_df)

                if ticker_df[ticker_df < 5].any().sum() > 0:
                    continue

                init_df = init_df.merge(ticker_df, left_index=True, right_index=True, how='outer')
            else:
                init_df = ticker_df
                init_df = init_df.add_suffix('__' + ticker[:-4])
            print(init_df)
            count += 1
            # if count > 10:
            #     break


        # init_df = init_df.mask(init_df < 5)

        #try:
         #   init_df = init_df[start_date:end_date]
        #except:

        # not sure these two steps dates and columns which one first is better?
        # print('init_df bf idx', init_df)
        all_useful_dates = []
        all_useless_idx = []
        for idx, x in init_df.iterrows():
            cur_date = datetime.strptime(idx, '%Y-%m-%d')
            if init_df.loc[[idx]].isna().sum().sum() / len(init_df.columns) > 0.3:
                all_useless_idx.append(idx)
            else:
                all_useful_dates.append(idx)

        init_df = init_df.drop(index=all_useless_idx)

        # print('init_df', init_df)
        all_useful_tickers = []
        for column in init_df.columns:
            if init_df[column].isna().sum() / len(init_df) < 0.02:
                all_useful_tickers.append(column.split('__')[-1])

        all_useful_tickers = list(set(all_useful_tickers))
        print(all_useful_tickers)
        return all_useful_dates, all_useful_tickers

    '''
        Transform the original EOD data collected from Google Finance to a
        friendly format to fit machine learning model via the following steps:
            Calculate moving average (5-days, 10-days, 20-days, 30-days), etc.
            ignoring suspension days (market open, only suspend this stock)
            Normalize features by (feature - min) / (max - min)
    '''

    def generate_feature(self, if_date_list, selected_tickers_fname, begin_date, end_date,
                         return_days=1, pad_begin=29):
        self._get_market_dir(os.path.join(self.data_path))

        if if_date_list:            
            # load from a list of dates to exclude holidays
            trading_dates = np.genfromtxt(
                os.path.join(self.data_path, '..',
                             self.market_name + '_aver_line_dates.csv'),
                dtype=str, delimiter=',', skip_header=False
            )
        else:
            trading_dates, self.tickers = self._get_dates(os.path.join(self.data_path), begin_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        print('#trading dates:', len(trading_dates))
        print('begin date:', begin_date)

        # make a dictionary to translate dates into index and vice versa
        index_tra_dates = {}
        tra_dates_index = {}
        for index, date in enumerate(trading_dates):
            tra_dates_index[date] = index
            index_tra_dates[index] = date

        if selected_tickers_fname:
            self.tickers = np.genfromtxt(
                os.path.join(self.data_path, '..', selected_tickers_fname),
                dtype=str, delimiter='\t', skip_header=False
            )

        print('#tickers selected:', len(self.tickers))
        self._read_EOD_data(self.data_path)
        stocks_features = []
        labels = []

        # load stock industry relation data
        # industry_ticker_file = os.path.join('data/relation/sector_industry/',
        #                                     self.market_name + '_industry_ticker.json')
        # ticker_index = {}
        # for index, ticker in enumerate(self.tickers):
        #     ticker_index[ticker] = index
        # with open(industry_ticker_file, 'r') as fin:
        #     industry_tickers = json.load(fin)
        # print('#industries: ', len(industry_tickers))

        # # concatenate lists of all tickers of all industry
        # all_tickers_from_industry = sum(industry_tickers.values(), [])
        # assert (set(self.tickers) == set(sum(industry_tickers.values(), [])))

        # # create indices for the industries
        # valid_industry_count = 0
        # valid_industry_index = {}
        # for industry in industry_tickers.keys():
        #     valid_industry_index[industry] = valid_industry_count
        #     valid_industry_count += 1

        # # assign the industry index to corresponding stocks of each industry
        # industry_all_tickers = {}
        # for industry in valid_industry_index.keys():
        #     cur_ind_tickers = industry_tickers[industry]
        #     ind_ind = valid_industry_index[industry]
        #     for i in range(len(cur_ind_tickers)):
        #         industry_all_tickers[cur_ind_tickers[i]] = ind_ind
        # assert (set(self.tickers) == set(list(industry_all_tickers.keys())))

        for index, ticker in enumerate(self.tickers):
            print('ticker: ', ticker)
            # industry_relation = industry_all_tickers[ticker]

            # the index of self.tickers and self.data_EOD matches.
            # So we get the index and then fetch EOD data.
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

            # print(begin_date_row)
            selected_EOD_str = single_EOD[begin_date_row:end_date_row]
            print(selected_EOD_str)
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
                if count_5 == 0:
                    mov_aver_features[row][0] = -1234
                else:
                    mov_aver_features[row][0] = aver_5 / count_5
                if count_10 == 0:
                    mov_aver_features[row][1] = -1234
                else:
                    mov_aver_features[row][1] = aver_10 / count_10
                mov_aver_features[row][2] = aver_20 / count_20
                mov_aver_features[row][3] = aver_30 / count_30
                if len(std_5) <= 1:
                    mov_aver_features[row][4] = -1234
                else:
                    mov_aver_features[row][4] = statistics.stdev(std_5)
                if len(std_10) <= 1:
                    mov_aver_features[row][5] = -1234
                else:
                    mov_aver_features[row][5] = statistics.stdev(std_10)
                mov_aver_features[row][6] = statistics.stdev(std_20)
                mov_aver_features[row][7] = statistics.stdev(std_30)

            # print('industry_relation:', industry_relation)

            # check abnormal values
            pri_min = np.min(selected_EOD[begin_date_row:, 4])
            price_max = np.max(selected_EOD[begin_date_row:, 4])

            if pri_min < 5:
                print('pri_min', pri_min)
                print('this stock is skipped')
                continue

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
                        if selected_EOD[row][-2] == 0: # selected_EOD[row + return_days][-2] == 0 or comment off because unlike log return raw return can be 0
                            features[cur_index - pad_begin][-1] = -1234
                        else:
                            features[cur_index - pad_begin][-1] = (selected_EOD[row + return_days][-2] - selected_EOD[row][-2]) / selected_EOD[row][-2]
                        if np.abs((selected_EOD[row + return_days][-2] - selected_EOD[row][-2])/selected_EOD[row][-2]) > 1:
                            print('np.abs(selected_EOD[row + return_days][-2] - selected_EOD[row][-2])!!!!!!!!!!!!!!', np.abs(selected_EOD[row + return_days][-2] - selected_EOD[row][-2]))
                            # print()
                            # sys.exit()

            '''normalize the log of volume feature'''
            features[:, -3][features[:, -3] > 0] = np.log(features[:, -3][features[:, -3] > 0])

            # for the stock DWPP whose 0 is missing values
            if (features[:, :-1] == 0).any():
                features[:, :-1][features[:, :-1] == 0] = -1234
                print('np.sum(list(features == 5))', np.sum(list(features == -1234)))

            # features[:, -2] = industry_relation
            features[:, -2] = 0

            # last row all missing values
            features = np.delete(features, -1, 0)
            # because the last row don't have label since no tomorrow data
            features = np.delete(features, -1, 0)

            labels.append(features[:, -1:])
            for j in range(len(features[:, -1])):
                if np.abs(features[:, -1][j]) > 2 and np.abs(features[:, -1][j]) != 1234:
                    print('> 2!!!!!!!!!!', features[:, -1][j])
                    print('index', j)
                    print('features[j-1]', features[j-1])
                    print('features[j]', features[j])
                    print('features[j+1]', features[j+1])
                    print(self.tickers[index])
                    # sys.exit()

            # normalization
            features = np.delete(features, 14, 1)
            max_num = np.max(features[:, :4][features[:, :4] != -1234])
            min_num = np.min(features[:, :4][features[:, :4] != -1234])
            max_num2 = np.max(features[:, 8:12][features[:, 8:12] != -1234])
            min_num2 = np.min(features[:, 8:12][features[:, 8:12] != -1234])
            max_vol = np.max(features[:, 4:8][features[:, 4:8] != -1234])
            min_vol = np.min(features[:, 4:8][features[:, 4:8] != -1234])
            max_volume = np.max(features[:, 12][features[:, 12] != -1234])
            min_volume = np.min(features[:, 12][features[:, 12] != -1234])
            max_num = np.maximum(max_num2, max_num)
            min_num = np.minimum(min_num2, min_num)
            for i in range(np.shape(features)[1]):
                not_mask_column = features[:, i][features[:, i] != -1234]
                if i in [0, 1, 2, 3] or i in [8, 9, 10, 11]:
                    features[:, i][features[:, i] != -1234] = (not_mask_column - min_num)/(max_num - min_num)
                elif i in [4, 5, 6, 7]:
                    features[:, i][features[:, i] != -1234] = (not_mask_column - min_vol)/(max_vol - min_vol)
                elif i in [12]:
                    features[:, i][features[:, i] != -1234] = (not_mask_column - min_volume) / (max_volume - min_volume)
            # if np.shape(features)[0] != 1244:
            #     sys.exit()

            assert (sum(list(features[:, 13] == -1234)) == 0)
            assert (sum(list(features[:, -1] == -1234)) == 0)
            stocks_features.append(features)
        for i in range(len(labels)):
            for j in range(len(labels[i])):
                if np.abs(labels[i][j]) > 1 and np.abs(labels[i][j]) != 1234:
                    print('> 1!!!!!!!!!!', labels[i][j])
                    print('index', j)
                    print(self.tickers[i])
                    # sys.exit()
        
        # stack_labels = np.vstack(labels) # not vstack because the vector is no longer vertical
        # stack_stocks_features = np.vstack(stocks_features)
        np.save(self.market_name+'_stocks_features.npy', stocks_features)
        np.save(self.market_name+'_labels.npy', labels)
        # X_train, X_test, y_train, y_test = train_test_split(stocks_features, labels, test_size=0.33, random_state=42, shuffle=False)
        return stocks_features, labels


def create_dataset(
    dataset_name, stock_tasks,
    num_train_examples, num_valid_examples, num_test_examples,
    seed, load_fn):
    """Create a dataset from the given spec and seed."""

    pos = stock_tasks

    data, labels = get_dataset(
        dataset_name,
        pos, load_fn=load_fn) # didn't use the second argument to limit the sample size of pos/neg class
    # labels = labels[np.in1d(labels[:, 0], pos)] # from first column axis0 find pos and neg index and return the labels
    # labels = np.delete(labels, 0, 1) # delete one dimension because the first dimension is the number I set to get pairs, e.g. 801 and 800

    (train_data, train_labels, valid_data, valid_labels,
     test_data, test_labels) = train_valid_test_split(
         data, labels,
         num_train_examples,
         num_valid_examples,
         num_test_examples,
         seed)

    print('valid_data', valid_data.shape)
    print('num_valid_examples', num_valid_examples)

    dataset = task_pb2.ScalarLabelDataset()
    for i in range(train_data.shape[0]):
        train_feature = dataset.train_features.add()
        train_feature.features.extend(list(train_data[i]))
        dataset.train_labels.append(train_labels[i][0])
        if np.abs(train_labels[i]) > 2 and np.abs(train_labels[i]) != 1234:
            print('train_labels[i]!!!!!!!!!!!!!!',
                  train_labels[i])
            # sys.exit()
    print(valid_data.shape)
    for i in range(valid_data.shape[0]):
        valid_feature = dataset.valid_features.add()
        valid_feature.features.extend(list(valid_data[i]))
        dataset.valid_labels.append(valid_labels[i][0])
        if np.abs(valid_labels[i]) > 2 and np.abs(valid_labels[i]) != 1234:
            print('valid_labels[i]!!!!!!!!!!!!!!',
                  valid_labels[i])
            # sys.exit()
    # print('dataset.valid_features_size()', dataset.valid_features_size())
    # sys.exit()
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
    name, stock_tasks=None, load_fn=None):

    # Load datasets.
    dataset_dict = load_fn(
        name)

    train_data = dataset_dict['features']
    train_labels = dataset_dict['labels']

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
    ''' change stratefy=False in the context of changing lalebs into values and coding financial measures'''
    if num_test_examples > 0:
        if use_stratify:
            stratify = labels
        else:
            stratify = None
        train_data, test_data, train_labels, test_labels = (
            sklearn.model_selection.train_test_split(
                data, labels,
                train_size=(
                    num_train_examples +
                    num_valid_examples),
                test_size=num_test_examples,
                random_state=seed, shuffle=False, stratify=stratify))

    else:
        train_data, train_labels = data, labels
        test_data = None
        test_labels = None
    if use_stratify:
        stratify = train_labels
    else:
        stratify = None

    train_data, valid_data, train_labels, valid_labels = (
        sklearn.model_selection.train_test_split(
            train_data, train_labels,
            train_size=num_train_examples,
            test_size=num_valid_examples,
            random_state=seed, shuffle=False, stratify=stratify))
    return (
        train_data, train_labels,
        valid_data, valid_labels,
        test_data, test_labels)


flags.DEFINE_string(
    'data_dir', 'data_for_EvoAlgo_kdd2023_preds',
    'Path of the folder to save the datasets.')

flags.DEFINE_string(
    'path', 'data/eikon_data',
    'Path of EOD data.')

flags.DEFINE_string(
    'market', 'NASDAQ',
    'Market name.')

flags.DEFINE_integer('num_train_examples', 13,
                     'Number of training examples in each dataset.')

flags.DEFINE_integer('num_valid_examples', 1216,
                     'Number of validation examples in each dataset.')

# set to 0 here because we split num_valid_examples 244 into valid and test later
flags.DEFINE_integer('num_test_examples', 0,
                     'Number of test examples in each dataset.')

flags.DEFINE_integer('projected_dim', 13,
                     'The dimensionality to project the data into.')

flags.DEFINE_string('dataset_name', 'NASDAQ',
                    'Name of the dataset.')

flags.DEFINE_string('start_date', '2018-1-2',
                    'start_date.')

flags.DEFINE_string('end_date', '2022-12-31',
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

flags.DEFINE_list('stock_tasks', None,
                  'Stock tasks included.')

FLAGS = flags.FLAGS


def main(argv):
    """Create and save the datasets."""
    processor = EOD_Preprocessor(FLAGS.path, FLAGS.market)

    ticker_list = ''
    if FLAGS.if_ticker_list:
        ticker_list = processor.market_name + '_tickers_qualify_dr-0.98_min-5_smooth.csv'

    len_tickers = 0
    if os.path.isfile(processor.market_name+'_stocks_features.npy') and os.path.isfile(processor.market_name+'_labels.npy'):
        stock_features = np.load(processor.market_name+'_stocks_features.npy')
        stock_labels = np.load(processor.market_name+'_labels.npy')
        print('len of stock_features', len(stock_features))
        print('stock_features', np.shape(stock_features[0]))
        FLAGS.num_valid_examples = np.shape(stock_features[0])[0] - FLAGS.num_train_examples
        len_tickers = len(stock_features)
    else:
        stock_features, stock_labels = processor.generate_feature(
            FLAGS.if_date_list,
            ticker_list,
            datetime.strptime(FLAGS.start_date, processor.date_format),
            datetime.strptime(FLAGS.end_date, processor.date_format),
            return_days=1,
            pad_begin=29
        )
        len_tickers = len(processor.tickers)

    FLAGS.stock_tasks = list(range(0, len_tickers))

    if not os.path.exists(FLAGS.data_dir + FLAGS.market):
        os.makedirs(FLAGS.data_dir + FLAGS.market)

    FLAGS.data_dir = FLAGS.data_dir + FLAGS.market

    name = processor.market_name
    dataset_dict = {'features': stock_features, 'labels': stock_labels}
    dataset_dict[name] = dataset_dict

    def load_fn(name):
        return dataset_dict[name]

    # stock_tasks = sorted([int(x) for x in FLAGS.stock_tasks])
    for stock_tasks in FLAGS.stock_tasks:
        print('Generating stock_task {}'.format(stock_tasks))

        random_seeds = range(FLAGS.min_data_seed, FLAGS.max_data_seed)
        for seed in random_seeds:
            dataset = create_dataset(
                processor.market_name, stock_tasks,
                FLAGS.num_train_examples, FLAGS.num_valid_examples,
                FLAGS.num_test_examples, seed, load_fn)
            filename = 'dataset_{}-stock_{}-dim_{}-seed_{}'.format(
                processor.market_name, stock_tasks,
                FLAGS.projected_dim, seed)
            serialized_dataset = dataset.SerializeToString()

            with open(os.path.join(FLAGS.data_dir, filename), 'wb') as f:
                f.write(serialized_dataset)


if __name__ == '__main__':
    app.run(main)
