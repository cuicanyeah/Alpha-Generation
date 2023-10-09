# import required module
import os
import argparse
import shutil
import random
from glob import glob
import sys
import pandas as pd
import numpy as np
# import libraries
# from libraries.settings import *
from scipy.stats.stats import pearsonr
import itertools

parser = argparse.ArgumentParser("sota")
parser.add_argument('-n', type=int, default=10, help='number of alphas')
parser.add_argument('-p', help='output path',
                    default='data/2013-01-01')
parser.add_argument('-m', type=str, default='_nyse_', help='alpha data path')
parser.add_argument('-t', type=float, default=0.028, help='alpha selection threshold')
parser.add_argument('-p_list', '--names_list', nargs='+', default=[])
parser.add_argument('-i', type=int, default=0, help='include best')

args = parser.parse_args()

if not os.path.exists(args.p):
    os.makedirs(args.p)

# assign directory
directory = args.p

# iterate over files in
# that directory
all_alphas = []
# for filename in os.listdir(directory):
# 	f = os.path.join(directory, filename)
	# checking if it is a file

best_ic = 0
best_alpha_name = ''

alphas_returns = {}
f_alphas_returns = {}
num_alphas = 0
for dir_name in args.names_list:
	for path in os.listdir(dir_name):
		if args.m in path and '_nasdaq_3' not in path:
			print(path)
			f = os.path.join(dir_name, path, 'performance.txt')
			if os.path.isfile(f):
				num_alphas += 1
				print(f)
				with open(f) as file:
					lines = file.readlines()
					# print(lines[-1])
					all_returns = lines[1].split('strategy_returns')[1:]
					for strategy_return_str in all_returns:
						split_str = strategy_return_str.split('[')
						strategy_return_str_list = split_str[1].split(']')[0]
						name_prefix = path.split('"')[0]
						alphas_returns[name_prefix+split_str[0]] = [float(i) for i in strategy_return_str_list.split(',')]					
			f = os.path.join(dir_name, path, 'log.txt')
			if os.path.isfile(f):
				print(f)
				with open(f) as file:
					lines = file.readlines()
					count = 0
					f_perf = ''
					for line in lines:
						if 'sym_performance' in line:
							if count != 1:
								count += 1
								continue
							else:
								f_perf = line

					print(f_perf)
					all_returns = f_perf.split('strategy_returns')[1:]
					print(all_returns)
					for strategy_return_str in all_returns:
						split_str = strategy_return_str.split('[')
						strategy_return_str_list = split_str[1].split(']')[0]
						name_prefix = path.split('"')[0]
						f_alphas_returns[name_prefix+split_str[0]] = [float(i) for i in strategy_return_str_list.split(',')]
						print('path+split_str[0]', path+split_str[0])
			else:
				print('this path no performance {}'.format(path))

			# else:
			# 	print('this path no performance {}'.format(path))				

print('num_alphas', num_alphas)
print(len(alphas_returns))
print(len(f_alphas_returns))
assert len(alphas_returns) == num_alphas * 6
assert len(alphas_returns) == len(f_alphas_returns)

def check_corr(selected_returns, selected_returns_names):
	input_data = np.transpose(np.asarray(selected_returns))
	df = pd.DataFrame(input_data, columns=[selected_returns_names]) 
	# print(df)

	#      gene_a    gene_b    gene_c    gene_d    gene_e
	# 0  0.471257  0.854139  0.781204  0.678567  0.697993
	# 1  0.292909  0.046159  0.250902  0.064004  0.307537
	# 2  0.422265  0.646988  0.084983  0.822375  0.713397
	# 3  0.113963  0.016122  0.227566  0.206324  0.792048
	# 4  0.357331  0.980479  0.157124  0.560889  0.973161

	correlations = {}
	columns = df.columns.tolist()

	for col_a, col_b in itertools.combinations(columns, 2):
		print(str(col_a) + '__' + str(col_b))
		correlations[str(col_a) + '__' + str(col_b)] = np.corrcoef(df.loc[:, col_a], df.loc[:, col_b])[0,1]

	result = pd.DataFrame.from_dict(correlations, orient='index')
	result.columns = ['PCC']#, 'p-value']

	# print(result.sort_index())

	return result



	

# selected_returns = list(alphas_returns.values())
# selected_returns_names = list(alphas_returns.keys())

# f_selected_returns = list(f_alphas_returns.values())
# f_selected_returns_names = list(f_alphas_returns.keys())

# itertools.combinations
# itertools.permutations
smallest_corr = 99
for len_alphas in reversed(range(2, num_alphas+1)):
	for all_comb in itertools.combinations(list(range(num_alphas)), len_alphas):
		print('all_comb!!!!!!!', all_comb)
		for all_perm in itertools.permutations(all_comb):
			print('all_perm', all_perm)
			# perm_selected_returns = [selected_returns[index] for index in all_perm]
			# perm_selected_returns_names = [selected_returns_names[index] for index in all_perm]
			# perm_f_selected_returns = [f_selected_returns[index] for index in all_perm]
			# perm_f_selected_returns_names = [f_selected_returns_names[index] for index in all_perm]
			selected_returns = []
			selected_returns_names = []

			f_selected_returns = []
			f_selected_returns_names = []
			for i, j in zip(all_perm, [0,2,3,4,5]):
# 	# ind = random.sample(list(range(6)), 1)
# 	ind = [i+1]
# 	print(i*6)
# 	# print(alphas_returns.keys())
# 	# print(list(alphas_returns.keys()))
				selected_returns_names.append(list(alphas_returns.keys())[i*6:(i+1)*6][j])
				selected_returns.append(list(alphas_returns.values())[i*6:(i+1)*6][j])

				f_selected_returns_names.append(list(f_alphas_returns.keys())[i*6:(i+1)*6][j])
				f_selected_returns.append(list(f_alphas_returns.values())[i*6:(i+1)*6][j])		

			result = check_corr(selected_returns, selected_returns_names)

			srt_list = sorted(result['PCC'].tolist())
			# print(srt_list)
			# if np.mean(srt_list[:10]) < 0.15:
			# 	print(np.mean(srt_list[:10]))

			# if np.mean(srt_list[:3]) < 0.15:
			# 	print(np.mean(srt_list[:3]))

			if np.mean(result['PCC']) < smallest_corr:
				smallest_corr = np.mean(result['PCC'])
			print('smallest_corr',smallest_corr)
			if np.mean(result['PCC']) < 0.17:
				print('result[PCC]', result['PCC'])
				print(np.mean(result['PCC']))
				f_results = check_corr(f_selected_returns, f_selected_returns_names)
				print(f_results)
				# pd.to_csv(os.path.join(args.p, 'weak_corr.csv'), result)
				# pd.to_csv(os.path.join(args.p, 'f_a_weak_corr.csv'), f_results)
				# sys.exit()
				break


# selected_alphas = random.sample(all_alphas, args.n)

# for idx, alpha in enumerate(selected_alphas):
# 	new_alpha_name = str(idx) + 'int_alpha.txt'
# 	shutil.copyfile(alpha, os.path.join(args.ap, new_alpha_name))

# print('best_ic', best_ic)

# if args.i > 0: 
# 	shutil.copyfile(best_alpha_name, os.path.join(args.ap, 'alpha0.txt'))
# 	print('best is copied!')