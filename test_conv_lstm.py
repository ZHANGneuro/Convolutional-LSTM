
import numpy as np
from os import listdir
from matplotlib import image
import torch as pt
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import re
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# test
a_pool=  np.linspace(0, 5, 5, endpoint=False).astype(int)
abs_pool = [1,3,5,7,9,11,13]
# ks_pool = [1,3,5,7,9,11,13]
ks_pool = [1,2, 3,4,5,6,7,8,9,10, 11,12, 13]
image_dir = '/Users/bo/Documents/PycharmProjects/dingwei_relational_memory/images_identity/'
file_dir_list = np.array([image_dir + f for f in listdir(image_dir) if ".png" in f])
file_list = [int(re.search('images_identity/(.*).png', f).group(1)) for f in file_dir_list]
file_dir_list = file_dir_list[np.argsort(file_list)]

batch_size = 7
identity_seq = [0, 2, 1, 2, 0, 1, 2, 0, 1, 1, 2, 0]
num_image = len(identity_seq)
image_width = 90
image_height = 40
num_neuron = 12

x_seq_image = np.zeros((num_image, 3, image_height, image_width))
x_seq_label = identity_seq
for ith_ima in list(range(0, num_image)):
    x_seq_image[ith_ima, ...] = image.imread(file_dir_list[x_seq_label[ith_ima]]).reshape(3, image_height, image_width)

y_seq_label = x_seq_label[1:]
y_seq_label.append(x_seq_label[0])
y_seq_label = np.array(y_seq_label)

y_seq_image = np.zeros((num_image, 3, image_height, image_width))
for ith_ima in list(range(0, num_image)):
    y_seq_image[ith_ima, ...] = image.imread(file_dir_list[y_seq_label[ith_ima]]).reshape(3, image_height, image_width)

tensor_image_x = pt.Tensor(x_seq_image).type(pt.float)
tensor_image_y = pt.Tensor(y_seq_image).type(pt.float)
tensor_label_y = pt.Tensor(y_seq_label).type(pt.LongTensor)
custom_dataset = TensorDataset(tensor_image_x, tensor_label_y)
my_dataloader = DataLoader(custom_dataset, batch_size=batch_size, drop_last=True)

perf_array = np.zeros((len(a_pool)*len(ks_pool)*len(abs_pool), 4))
# perf_array = np.zeros((len(ks_pool)*len(abs_pool), 3))
counter = 0
for ith_a in a_pool:
    for ith_ks in ks_pool:
        for ith_abs in abs_pool:
            kernel_size=ith_ks
            num_abstract=ith_abs
            dim_input_x = image_width - kernel_size + 1
            dim_input_y = image_height - kernel_size + 1
            dim_input = dim_input_x * dim_input_y
            perf_array[counter, 0] = ith_ks
            perf_array[counter, 1] = ith_abs
            PATH = './weights/weight_conv_lstm_ks' + str(kernel_size) + '_numAbs' + str(num_abstract) + '_ith' + str(ith_a) + '.pth'
            class ConV_lstm_autoencoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv = nn.Conv2d(3, num_neuron, kernel_size)
                    self.fc1 = nn.Linear(dim_input * num_neuron, num_abstract)
                    self.lstm = nn.LSTM(input_size=num_abstract, hidden_size=num_abstract, num_layers=1, batch_first=True)
                    self.fc2 = nn.Linear(num_abstract, 3)

                def forward(self, x):
                    x = self.conv(x)
                    x = pt.flatten(x, 1)
                    x = self.fc1(x)
                    x, _ = self.lstm(x)
                    x = self.fc2(x)
                    return x

            my_net = ConV_lstm_autoencoder()
            my_net.load_state_dict(pt.load(PATH))

            correct = 0
            total = 0
            with pt.no_grad():
                for data in my_dataloader:
                    images, labels = data
                    outputs = my_net(images)
                    _, predicted = pt.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            perf_array[counter, 2] = 100 * correct // total
            perf_array[counter, 3] = ith_a
            counter = counter + 1
            print(f'Accuracy: {100 * correct // total} %')


# # plot
# import matplotlib as mpl
# import matplotlib
# import matplotlib.pyplot as plt
# plt.close('all')
# font = {'family': 'Arial',
#         'weight': 'normal',
#         'size': 15}
# matplotlib.rc('font', **font)
# color_map_control = mpl.colormaps['cool'].resampled(8)
# color_map = color_map_control(range(0, len(ks_pool)))
# fig = plt.figure(figsize=(7, 7))
# ax = fig.add_subplot()
# for ith in list(range(0, len(ks_pool))):
#
# for axis in ['top', 'bottom', 'left', 'right']:
#     ax.spines[axis].set_linewidth(2)
#     ax.spines[axis].set_color((0 / 255, 0 / 255, 0 / 255))
# plt.xlabel('#abstract')
# plt.ylabel('Accuracy')
# plt.legend(loc = 'right' )
# plt.show()


import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
abs_pool = [1,3,5,7,9,11,13]
ks_pool = [1,2, 3,4,5,6,7,8,9,10, 11,12, 13]
plt.close('all')
font = {'family': 'Arial',
        'weight': 'normal',
        'size': 20}
matplotlib.rc('font', **font)
fig = plt.figure(figsize=(8, 10))
gs = gridspec.GridSpec(4,2)
gs.update(left=0.15, right=0.95, top=0.95, bottom=0.1, wspace=0.5, hspace=1)
for ith_ks in list(range(0, len(ks_pool))):
    cur_ks = []
    cur_abs = []
    cur_value = []
    cur_sd = []
    plot_x = np.linspace(0, len(abs_pool), len(abs_pool), endpoint=False)
    for ith_abs in list(range(0, len(abs_pool))):
        cur_ks.append(ks_pool[ith_ks])
        cur_abs.append(abs_pool[ith_abs])
        temp = perf_array[np.where((perf_array[:, 0] == ks_pool[ith_ks]) & (perf_array[:, 1] == abs_pool[ith_abs]))[0], 2]
        cur_value.append(np.mean(temp))
        cur_sd.append(np.std(temp) / np.sqrt(len(a_pool)))
    ax = fig.add_subplot(gs[ith_ks])
    ax.plot(plot_x, cur_value, color='black')
    cur_abs = np.array(cur_abs)
    cur_value = np.array(cur_value)
    cur_sd = np.array(cur_sd)
    plt.fill_between(plot_x, cur_value - cur_sd, cur_value + cur_sd, color='red', alpha=0.3)
    ax.set_title('kernal size=' + str(ks_pool[ith_ks]))
    ax.set_xlabel('#abstract locs')
    ax.set_ylabel('Performance')
    ax.set_ylim([30, 100])
    for axis in ['bottom', 'left', 'top', 'right']:
        ax.spines[axis].set_linewidth(2)
    ax.set_xticks(plot_x,abs_pool)
    # ax.axis('off')
plt.show()










import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
analysis_array = np.zeros((len(a_pool), len(ks_pool)))
anova_ks = []
anova_value = []
for ith_a in list(range(0, len(a_pool))):
    for ith_ks in list(range(0, len(ks_pool))):
        temp = perf_array[np.where((perf_array[:, 0] == ks_pool[ith_ks]) & (perf_array[:, 3] == a_pool[ith_a]))[0], 2]
        analysis_array[ith_a, ith_ks] = np.mean(temp)
        anova_ks.append(ks_pool[ith_ks])
        anova_value.append(np.mean(temp))
plot_mean = np.mean(analysis_array, axis=0)
plot_sd = np.std(analysis_array, axis=0)/ np.sqrt(len(a_pool))

plt.close('all')
font = {'family': 'Arial',
        'weight': 'normal',
        'size': 40}
matplotlib.rc('font', **font)
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(1,1)
ax = fig.add_subplot()
ax.plot(range(0, len(plot_mean)), plot_mean, color = 'black', linewidth = 5)
ax.fill_between(range(0, len(plot_mean)), plot_mean - plot_sd, plot_mean + plot_sd, color='red', alpha=0.3)
ax.set_xlabel('Kernal size')
ax.set_ylabel('Performance')
for axis in ['bottom', 'left', 'top', 'right']:
    ax.spines[axis].set_linewidth(2)
ax.set_xticks(np.linspace(0, len(ks_pool), len(ks_pool), endpoint=False),ks_pool)
plt.subplots_adjust(left=0.25, right=1, top=0.95, bottom=0.15)
plt.show()

d = pd.DataFrame()
d["ks"] = anova_ks
d["value"] = anova_value
model = ols("value ~ ks",data = d).fit()
anova_table = sm.stats.anova_lm(model, typ=2)


