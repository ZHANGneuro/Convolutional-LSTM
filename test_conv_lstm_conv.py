

import numpy as np
from os import listdir
import re
import matplotlib.pyplot as plt
from matplotlib import image
import torch as pt
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.patches as patches
from matplotlib import gridspec
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

image_dir = '/Users/bo/Documents/PycharmProjects/dingwei_relational_memory/images_identity/'
file_dir_list = np.array([image_dir + f for f in listdir(image_dir) if ".png" in f])
file_list = [int(re.search('images_identity/(.*).png', f).group(1)) for f in file_dir_list]
file_dir_list = file_dir_list[np.argsort(file_list)]

identity_seq = [0, 2, 1, 2, 0, 1, 2, 0, 1, 1, 2, 0]

batch_size = 7
image_width = 90
image_height = 40
num_neuron = 12
num_image = len(identity_seq)
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
custom_dataset = TensorDataset(tensor_image_x, tensor_image_y)
my_dataloader = DataLoader(custom_dataset, batch_size=batch_size, drop_last=True)


# test
a_pool=  np.linspace(0, 1, 1, endpoint=False).astype(int)
abs_pool = [7]
ks_pool = [1, 3, 5, 7, 9, 11, 13]
# perf_array = np.zeros((len(a_pool) * len(ks_pool)*len(abs_pool), 4))
perf_array = np.zeros((len(ks_pool)*len(abs_pool), 4))
counter = 0

for ith_a in list(range(0, 1)):
    for ith_ks in ks_pool:
        for ith_abs in abs_pool:
            kernel_size = ith_ks
            num_abstract = ith_abs
            dim_input_x = image_width - kernel_size + 1
            dim_input_y = image_height - kernel_size + 1
            dim_input = dim_input_x * dim_input_y
            perf_array[counter, 0] = ith_ks
            perf_array[counter, 1] = ith_abs
            perf_array[counter, 2] = ith_a
            class ConV_lstm_autoencoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv = nn.Conv2d(3, num_neuron, kernel_size)
                    self.fc1 = nn.Linear(dim_input * num_neuron, num_abstract)
                    self.lstm = nn.LSTM(input_size=num_abstract, hidden_size=num_abstract, num_layers=1, batch_first=True)
                    self.fc2 = nn.Linear(num_abstract, dim_input * num_neuron)
                    self.t_conv = nn.ConvTranspose2d(num_neuron, 3, kernel_size)
                def forward(self, x):
                    x = F.relu(self.conv(x))
                    x = pt.flatten(x, 1)
                    x = F.relu(self.fc1(x))
                    x, _ = self.lstm(x)
                    x = F.relu(self.fc2(x))
                    x = x.reshape(batch_size, num_neuron, dim_input_y, dim_input_x)
                    x = F.sigmoid(self.t_conv(x))
                    return x

            my_net = ConV_lstm_autoencoder()
            PATH = './weights/weight_conv_lstm_deconv_ks'+str(kernel_size)+'_numAbs'+str(num_abstract) + '_a' + str(ith_a) +'.pth'
            my_net.load_state_dict(pt.load(PATH))

            with pt.no_grad():
                # x seq [0, 2, 1, 2, 0, 0, 1, 2, 2, 0]
                num_corr = 0
                for ith_attempt in list(range(0, 10)):
                    ref_index = random.choice(list(range(0, len(identity_seq)-1-batch_size)))
                    cur_x_segment = []
                    cur_y_segment = []
                    for ith in list(range(0, batch_size)):
                        cur_x_segment.append(x_seq_label[ref_index + ith])
                        cur_y_segment.append(y_seq_label[ref_index + ith])
                    input_ima = np.zeros((batch_size, 3, image_height, image_width))
                    corr_ima = np.zeros((batch_size, 3, image_height, image_width))
                    for ith in list(range(0, len(cur_x_segment))):
                        input_ima[ith, ...] = image.imread(file_dir_list[cur_x_segment[ith]]).reshape(3, image_height, image_width)
                        corr_ima[ith, ...] = image.imread(file_dir_list[cur_y_segment[ith]]).reshape(3, image_height, image_width)
                    outputs = my_net(pt.Tensor(input_ima).type(pt.float))
                    outputs = outputs.reshape(batch_size, image_height, image_width, 3).detach().numpy()

                    for ith in list(range(0, batch_size)):
                        cur_pred = np.mean(outputs[ith], axis=2)
                        index_pred = np.argmin([np.mean(cur_pred[:, 0:30]), np.mean(cur_pred[:, 31:60]), np.mean(cur_pred[:, 61:90])])
                        if (index_pred == cur_y_segment[ith]):
                                num_corr = num_corr + 1
                        plt.close('all')

                perf_array[counter, 3] = num_corr/70
            counter = counter + 1
            print(str(ith_a)+ '-' + str(ith_ks) + '-' + str(ith_abs))

# np.save('/Users/bo/Documents/PycharmProjects/dingwei_relational_memory/model_performance.npy', perf_array)






# # one way anova
# import pandas as pd
# import statsmodels.api as sm
# import matplotlib
# from statsmodels.formula.api import ols
# analysis_array = np.zeros((len(a_pool), len(ks_pool)))
# anova_ks = []
# anova_value = []
# for ith_a in list(range(0, len(a_pool))):
#     for ith_ks in list(range(0, len(ks_pool))):
#         temp = perf_array[np.where((perf_array[:, 0] == ks_pool[ith_ks]) & (perf_array[:, 2] == a_pool[ith_a]))[0], 3]
#         analysis_array[ith_a, ith_ks] = np.mean(temp)
#         anova_ks.append(ks_pool[ith_ks])
#         anova_value.append(np.mean(temp))
# plot_mean = np.mean(analysis_array, axis=0)
# plot_sd = np.std(analysis_array, axis=0)/ np.sqrt(len(a_pool))
#
# plt.close('all')
# font = {'family': 'Arial',
#         'weight': 'normal',
#         'size': 40}
# matplotlib.rc('font', **font)
# fig = plt.figure(figsize=(8, 8))
# gs = gridspec.GridSpec(1,1)
# ax = fig.add_subplot()
# ax.plot(range(0, len(plot_mean)), plot_mean, color = 'black', linewidth = 5)
# ax.fill_between(range(0, len(plot_mean)), plot_mean - plot_sd, plot_mean + plot_sd, color='red', alpha=0.3)
# ax.set_xlabel('Kernal size')
# ax.set_ylabel('Performance')
# # ax.set_ylim([0.2, 0.9])
# for axis in ['bottom', 'left', 'top', 'right']:
#     ax.spines[axis].set_linewidth(2)
# ax.set_xticks(np.linspace(0, len(ks_pool), len(ks_pool), endpoint=False),ks_pool)
# plt.subplots_adjust(left=0.25, right=1, top=0.95, bottom=0.15)
# plt.show()
#
# d = pd.DataFrame()
# d["ks"] = anova_ks
# d["value"] = anova_value
# model = ols("value ~ ks",data = d).fit()
# anova_table = sm.stats.anova_lm(model, typ=2)
#
#
# # two way anova
# import pandas as pd
# import statsmodels.api as sm
# from statsmodels.formula.api import ols
# anova_ks = []
# anova_abs = []
# anova_value = []
# anova_sd = []
# for ith_ks in list(range(0, len(ks_pool))):
#     for ith_abs in list(range(0, len(abs_pool))):
#         temp = perf_array[np.where( (perf_array[:, 0] == ks_pool[ith_ks]) & (perf_array[:, 1] == abs_pool[ith_abs]) )[0], 3]
#         anova_ks.append(ith_ks)
#         anova_abs.append(ith_abs)
#         anova_value.append(np.mean(temp))
#         anova_sd.append(np.std(temp)/ np.sqrt(len(temp)))
#
# d = pd.DataFrame()
# d["ks"] = anova_ks
# d["abs"] = anova_abs
# d["value"] = anova_value
# model = ols("value ~ ks*abs",data = d).fit()
# anova_table = sm.stats.anova_lm(model, typ=2)
#
#
#
# # in one plot
# import matplotlib.pyplot as plt
# from matplotlib import gridspec
# import matplotlib
# import pandas as pd
# import statsmodels.api as sm
# from statsmodels.formula.api import ols
# import numpy as np
# from matplotlib.cm import ScalarMappable
# abs_pool = [1, 3, 5, 7, 9, 11, 13]
# ks_pool = [1,2, 3,4,5,6,7,8,9,10, 11,12, 13]
# colormap = plt.get_cmap('cool')
# colors = colormap(np.linspace(0, 1, len(ks_pool)))
# plt.close('all')
# font = {'family': 'Arial',
#         'weight': 'normal',
#         'size': 45}
# matplotlib.rc('font', **font)
# fig = plt.figure(figsize=(11, 8))
# gs = gridspec.GridSpec(1, 1)
# ax = fig.add_subplot()
# ks_mean = []
# for ith_ks in list(range(0, len(ks_pool))):
#     cur_ks = []
#     cur_abs = []
#     cur_value = []
#     cur_sd = []
#     for ith_abs in list(range(0, len(abs_pool))):
#         cur_ks.append(ks_pool[ith_ks])
#         cur_abs.append(abs_pool[ith_abs])
#         temp = perf_array[
#             np.where((perf_array[:, 0] == ks_pool[ith_ks]) & (perf_array[:, 1] == abs_pool[ith_abs]))[0], 3]
#         cur_value.append(np.mean(temp))
#         cur_sd.append(np.std(temp) / np.sqrt(len(a_pool)))
#     ks_mean.append(np.mean(cur_value))
#     plot_ax = ax.plot(cur_abs, cur_value, color=colors[ith_ks])
#     ax.fill_between(cur_abs, np.array(cur_value)-np.array(cur_sd), np.array(cur_value) + np.array(cur_sd), color=colors[ith_ks], alpha=0.3)
# ax.set_xlabel('#abstract')
# ax.set_ylabel('Performance')
# ax.set_title('Visual-Action model')
# # ax.set_ylim([20, 100])
# for axis in ['bottom', 'left', 'top', 'right']:
#     ax.spines[axis].set_linewidth(2)
# ax.set_xticks(abs_pool, abs_pool)
# plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
# plt.colorbar(ScalarMappable(cmap=colormap, norm=plt.Normalize(1, 13)), ticks=np.array(ks_pool), label='Kernel size')
# plt.show()



