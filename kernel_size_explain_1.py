import numpy as np
from os import listdir
import re
from matplotlib import image
import torch as pt
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
import random

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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
ks_pool = [1, 3, 5, 7, 9, 11, 13]
output_ima = np.zeros((40, 90, len(ks_pool)))
for ith_ks in list(range(0, len(ks_pool))):
    kernel_size = ks_pool[ith_ks]

    a_pool = np.linspace(0, 1, 1, endpoint=False).astype(int)
    abs_pool = [7]
    counter = 0

    num_abstract = 7
    dim_input_x = image_width - kernel_size + 1
    dim_input_y = image_height - kernel_size + 1
    dim_input = dim_input_x * dim_input_y

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
    PATH = './weights/weight_conv_lstm_deconv_ks' + str(kernel_size) + '_numAbs' + str(num_abstract) + '_a0' + '.pth'
    dict = pt.load(PATH)
    my_net.load_state_dict(dict)

    # test dataset
    cur_x_segment = []
    cur_y_segment = []
    for ith in list(range(0, batch_size)):
        cur_x_segment.append(x_seq_label[ith])
        cur_y_segment.append(y_seq_label[ith])
    input_ima = np.zeros((batch_size, 3, image_height, image_width))
    corr_ima = np.zeros((batch_size, 3, image_height, image_width))
    for ith in list(range(0, len(cur_x_segment))):
        input_ima[ith, ...] = image.imread(file_dir_list[cur_x_segment[ith]]).reshape(3, image_height, image_width)
        corr_ima[ith, ...] = image.imread(file_dir_list[cur_y_segment[ith]]).reshape(3, image_height, image_width)


    # test
    import matplotlib.pyplot as plt
    with pt.no_grad():
        outputs = my_net(pt.Tensor(input_ima).type(pt.float))
        plot_activity = outputs[3]
        plot_activity = plot_activity.reshape(40, 90, 3).detach().numpy()
        plot_activity = np.mean(plot_activity, axis=2)
        output_ima[..., ith_ks] = plot_activity
        # fig = plt.figure(figsize=(12, 8))
        # ax = fig.add_subplot()
        # xxx = ax.imshow(plot_activity, cmap='jet')
        # ax.axis('off')
        # plt.colorbar(xxx)
        # plt.show()


# plot multi
import matplotlib
import imageio as iio
from matplotlib import gridspec
plt.close('all')
font = {'family': 'Arial',
        'weight': 'normal',
        'size': 70}
matplotlib.rc('font', **font)
fig = plt.figure(figsize=(7, 15))
gs = gridspec.GridSpec(len(ks_pool), 1)
gs.update(left=0, right=1, top=1, bottom=0)
for ith in list(range(0, len(ks_pool))):
    ax1 = fig.add_subplot(gs[ith])
    ax1.imshow(output_ima[..., ith])
    # ax1.set_title('Stimulus ' + str(ith))
    ax1.axis('off')
plt.show()


# plot target
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot()
xxx = ax.imshow(corr_ima[3].reshape(40, 90, 3), cmap='jet')
for axis in ['top','right','bottom','left']:
    ax.spines[axis].set_linewidth(4)
plt.xticks([])
plt.yticks([])
plt.show()







#
# # feature map
# # cur_weight = dict['conv.weight'].detach().numpy()
# cur_weight = dict['t_conv.weight'].detach().numpy()
# cur_weight = cur_weight.reshape(36, 13, 13)
# print(cur_weight.shape)
# import matplotlib.pyplot as plt
# plt.figure(figsize=(6, 6))
# for ith in list(range(0, 36)):
#     plt.subplot(6, 6, ith+1) # we have 5x5 filters and total of 16 (see printed shapes)
#     plt.imshow(cur_weight[ith, :, :], cmap='jet')
#     plt.axis('off')
# plt.show()
#     # plt.savefig('filter1.png')
#
#
# # target map
# x = input_ima.reshape(batch_size, num_neuron, dim_input_y, dim_input_x)
# model_children = list(my_net.children())
# conv_layers = []
# for i in range(len(model_children)):
#     if type(model_children[i]) == nn.ConvTranspose2d:
#         print(i)
#         conv_layers.append(model_children[i])
# outputs = []
# names = []
# for layer in conv_layers:
#     image = layer(image)
#     outputs.append(image)
#     names.append(str(layer))
#
#
