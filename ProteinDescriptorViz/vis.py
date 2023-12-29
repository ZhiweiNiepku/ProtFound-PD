import pandas as pd
import hypertools as hyp
import matplotlib.pyplot as plt
# Load all csv files into a single pandas dataframe
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

import csv
import numpy as np
plt.style.use('classic')

def read_csv(file_name):
    with open(file_name, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        data = []
        for row in reader:
            row_data = []
            for i in range(6):
                if i!=2:
                    row_data.extend(eval(row[i]))
            data.append(row_data)
            if len(row_data)!= 8775:
                print(len(row_data))
    return data

names = ['FactorAnalysis']
colors = ['#9CD5DD','#ECE79C','#ED7B7C']
# read csv files
data =read_csv('batch1_44277_0.csv') + read_csv('batch1_44277_1.csv') + read_csv('batch1_44277_2.csv')  + read_csv('batch1_44277_3.csv') + read_csv('batch1_44277_4.csv')  + read_csv('batch1_6025_.csv') + read_csv('batch1_157_.csv')
data = np.array(data)
print(data.shape)
group = [3 for i in range(44277)] + [2 for i in range(6025)] + [1 for i in range(157)]

# Visualize the reduced data with color labels
for name in names:
    geo = hyp.plot(data, '.',reduce=name, ndims = 3,  color = colors, hue = group)
    ax1 = geo.ax
    plt.title('FactorAnalysis')
    plt.tight_layout()
    plt.savefig('visualization_all_' + name + '.png', dpi=300, )