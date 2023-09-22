import copy
import numpy as np
import os
from evaluator_long_short_portfolio_for_LSTM import evaluate_sr
import ast
import json

def load_EOD_data(path, market_name, tickers, steps=1, alpha_data=[]):
    eod_data = []
    masks = []
    ground_truth = []
    base_price = []

    min_len_features=10000000000

    # all_file_names=[]
    # res = alpha_data[2:].split(']')  
    print('alpha_data',alpha_data)
    file_names=[]
    for file_name in alpha_data[1:-1].split(','):
        file_names.append(file_name.strip())
    print(file_names)
    print('')

    # all_file_names.append(file_names)
    
    # for item in res[1:]:
    #     print(item)    
    #     if item != '':
    #         file_names=[]
    #         for file_name in item[3:].split(','):
    #             file_names.append(file_name.strip())
    #         print(file_names)
    #         print('')
    #         all_file_names.append(file_names)

    all_EOD_from_alphas=[]
    all_labels_from_alphas=[]

    for file in file_names:
        print(file)
        if 'pred' in file:
            print(file)
            all_EOD=[]
            with open(os.path.join(path, file)) as f:
                lines = f.readlines()
                for i in lines:
                    if i != '\n':
                        # print('i: ', i)
                        this_ticker=[]
                        for j in i.split():
                            if j != '': 
                                this_ticker.append(float(j))
                        # print('len(this_ticker)', len(this_ticker))
                        # assert(len(this_ticker) == 1206)
                        if len(this_ticker) < min_len_features:
                            print(min_len_features)
                            min_len_features = len(this_ticker)
                        all_EOD.append(this_ticker)
                        # print('!!!!!!!!')
                        # print('len(all_EOD)', len(all_EOD))
            all_EOD_from_alphas.append(all_EOD)
        if 'label' in file:
            all_labels=[]
            with open(os.path.join(path, file)) as f:
                lines = f.readlines()
                for i in lines:
                    if i != '\n':
                        # print('i: ', i)
                        this_ticker=[]
                        for j in i.split():
                            if j != '': 
                                this_ticker.append(float(j))
                        # print('len(this_ticker)', len(this_ticker))
                        # assert(len(this_ticker) == 1206)
                        all_labels.append(this_ticker)
                        # print('!!!!!!!!')
                        # print('len(all_EOD)', len(all_labels))
            all_labels_from_alphas.append(all_labels)

    for index, all_EOD in enumerate(all_EOD_from_alphas): # loop over different alpha data
        for ind, all_EOD in enumerate(all_EOD_from_alphas[index]): # loop over alpha data's each stock
            if len(all_EOD_from_alphas[index][ind]) > min_len_features:
                # print(len(all_EOD_from_alphas[index][ind]))
                all_EOD_from_alphas[index][ind] = all_EOD_from_alphas[index][ind][-min_len_features:]

    for index, all_labels in enumerate(all_labels_from_alphas): # loop over different alpha data
        for ind, all_labels in enumerate(all_labels_from_alphas[index]): # loop over alpha data's each stock
            if len(all_labels_from_alphas[index][ind]) > min_len_features:
                # print(len(all_labels_from_alphas[index][ind]))
                all_labels_from_alphas[index][ind] = all_labels_from_alphas[index][ind][-min_len_features:]

    all_eod_data=[]
    for all_EOD, all_labels in zip(all_EOD_from_alphas, all_labels_from_alphas):   
        # print('all_EOD', all_EOD)
        # print('all_labels', all_labels)
        for index, ticker in enumerate(all_EOD):
            begin_date_row = 19
            mov_aver_features = np.ones([len(ticker), 5],
                               dtype=float) * -1234
            for row in range(begin_date_row, mov_aver_features.shape[0]):
                if abs(ticker[row] + 1234) > 1e-8 and abs(all_labels[index][row] + 1234) > 1e-8: 
                    aver_5 = 0.0
                    aver_10 = 0.0
                    aver_15 = 0.0
                    aver_20 = 0.0
                    count_5 = 0
                    count_10 = 0
                    count_15 = 0
                    count_20 = 0
                    for offset in range(20):
                        if abs(ticker[row - offset] + 1234) > 1e-8: 
                            if count_5 < 5:
                                count_5 += 1
                                aver_5 += ticker[row - offset]
                            if count_10 < 10:
                                count_10 += 1
                                aver_10 += ticker[row - offset]
                            if count_15 < 15:
                                count_15 += 1
                                aver_15 += ticker[row - offset]
                            if count_20 < 20:
                                count_20 += 1
                                aver_20 += ticker[row - offset]
                    mov_aver_features[row][0] = aver_5 / count_5
                    mov_aver_features[row][1] = aver_10 / count_10
                    mov_aver_features[row][2] = aver_15 / count_15
                    mov_aver_features[row][3] = aver_20 / count_20
                    mov_aver_features[row][4] = ticker[row]
            if index == 0:
                eod_data = np.zeros([len(all_EOD), len(ticker),
                                     5], dtype=np.float32)
                masks = np.ones([len(all_EOD), len(ticker)],
                                dtype=np.float32)
                ground_truth = np.zeros([len(all_EOD), len(ticker)],
                                        dtype=np.float32)       
            for row in range(len(ticker)):
                if abs(mov_aver_features[row][-1] + 1234) < 1e-8:
                    masks[index][row] = 0.0
                elif row > steps - 1 and abs(mov_aver_features[row - steps][-1] + 1234) \
                        > 1e-8:
                    ground_truth[index][row] = \
                        all_labels[index][row]
                for col in range(mov_aver_features.shape[1]):
                    if abs(mov_aver_features[row][col] + 1234) < 1e-8:
                        mov_aver_features[row][col] = 1.1
            eod_data[index, :, :] = mov_aver_features
        print('eod_data_shape',eod_data.shape)
        print('eod_data[:, :, 4]',eod_data[:, :, 4])
        all_eod_data.append(eod_data[:, :, 4])
    # c = np.concatenate([aux[..., np.newaxis] for aux in sequence_of_arrays], axis=3)
    eod_data=np.concatenate([aux[..., np.newaxis] for aux in all_eod_data], axis=2)
    print('eod_data_shape',eod_data.shape)
    print('ground_truth[:, -118:-2]', ground_truth[:, -116:])
    for i in range(len(all_eod_data)):
        print('eod_data[:, -116:, i]', eod_data[:, -116:, i])
        original_ic = evaluate_sr(eod_data[:, -116:, i], ground_truth[:, -116:], masks[:, -116:], False)
        print('original_ic: ', original_ic)
        original_ic = evaluate_sr(eod_data[:, -110:, i], ground_truth[:, -110:], masks[:, -110:], False)
        print('original_ic: ', original_ic)
    return eod_data, masks, ground_truth, len(all_eod_data)


def load_graph_relation_data(relation_file, lap=False):
    relation_encoding = np.load(relation_file)
    print('relation encoding shape:', relation_encoding.shape)
    rel_shape = [relation_encoding.shape[0], relation_encoding.shape[1]]
    mask_flags = np.equal(np.zeros(rel_shape, dtype=int),
                          np.sum(relation_encoding, axis=2))
    ajacent = np.where(mask_flags, np.zeros(rel_shape, dtype=float),
                       np.ones(rel_shape, dtype=float))
    degree = np.sum(ajacent, axis=0)
    for i in range(len(degree)):
        degree[i] = 1.0 / degree[i]
    np.sqrt(degree, degree)
    deg_neg_half_power = np.diag(degree)
    if lap:
        return np.identity(ajacent.shape[0], dtype=float) - np.dot(
            np.dot(deg_neg_half_power, ajacent), deg_neg_half_power)
    else:
        return np.dot(np.dot(deg_neg_half_power, ajacent), deg_neg_half_power)


def load_relation_data(relation_file):
    relation_encoding = np.load(relation_file)
    print('relation encoding shape:', relation_encoding.shape)
    rel_shape = [relation_encoding.shape[0], relation_encoding.shape[1]]
    mask_flags = np.equal(np.zeros(rel_shape, dtype=int),
                          np.sum(relation_encoding, axis=2))
    mask = np.where(mask_flags, np.ones(rel_shape) * -1e9, np.zeros(rel_shape))
    return relation_encoding, mask


def build_SFM_data(data_path, market_name, tickers):
    eod_data = []
    for index, ticker in enumerate(tickers):
        single_EOD = np.genfromtxt(
            os.path.join(data_path, market_name + '_' + ticker + '_1.csv'),
            dtype=np.float32, delimiter=',', skip_header=False
        )
        if index == 0:
            print('single EOD data shape:', single_EOD.shape)
            eod_data = np.zeros([len(tickers), single_EOD.shape[0]],
                                dtype=np.float32)

        for row in range(single_EOD.shape[0]):
            if abs(single_EOD[row][-1] + 1234) < 1e-8:
                # handle missing data
                if row < 3:
                    # eod_data[index, row] = 0.0
                    for i in range(row + 1, single_EOD.shape[0]):
                        if abs(single_EOD[i][-1] + 1234) > 1e-8:
                            eod_data[index][row] = single_EOD[i][-1]
                            # print(index, row, i, eod_data[index][row])
                            break
                else:
                    eod_data[index][row] = np.sum(
                        eod_data[index, row - 3:row]) / 3
                    # print(index, row, eod_data[index][row])
            else:
                eod_data[index][row] = single_EOD[row][-1]
        # print('test point')
    np.save(market_name + '_sfm_data', eod_data)