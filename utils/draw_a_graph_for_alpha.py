from absl import flags
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import re
import matplotlib.pyplot as plt
import sys
from absl import app

flags.DEFINE_string('path', "",
                     'Path to alpha performances to draw figure.')

flags.DEFINE_string('name', "",
                  'Name of the figure generated.')

FLAGS = flags.FLAGS

def main(argv):
    plt.rcParams.update({'font.size': 11})

    path = FLAGS.path

    save_all_curves_data = []
    list_of_all_sharpe = []
    list_of_num = []
    add = 0
    
    num = 0

    file_list = [f for f in listdir(path) if isfile(join(path, f))]

    count = 0

    for file in file_list:
        if 'Perf' in file:        
            list_of_num.append(int(re.findall("\d+", file)[num]) + add) 
            f = open(path + '/' + file, 'r')
            x = f.read().split(',')
            f.close()

            for item in x:
                if 'IC=' in item and 'test' in item:
                    list_of_all_sharpe.append(float(re.findall("\d+\.\d+", item)[0]))
    plt.figure()

    fig, ax = plt.subplots()
    ax.set_xticks([0,200000])
    plt.locator_params(axis='x', nbins=4)
    ax.plot(list_of_num, list_of_all_sharpe, 'o') 
    plt.xlabel('Number of searched alphas', fontsize=11) 
    plt.ticklabel_format(style='plain')
    plt.ylabel('IC', fontsize=11) 

    plt.show()
    figure_name = FLAGS.name+'.png'
    plt.savefig(figure_name)

if __name__ == '__main__':
    app.run(main)
