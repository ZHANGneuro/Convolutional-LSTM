# Convolutional LSTM
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

fig = plt.figure(figsize=(11, 15))
gs = gridspec.GridSpec(1, 1)
ax1 = fig.add_subplot(gs[0])
ax1.imshow(cur_pred)
plt.show()




image_dir = '/Users/bo/Documents/PycharmProjects/dingwei_relational_memory/images_identity/'
file_dir_list = np.array([image_dir + f for f in listdir(image_dir) if ".png" in f])
file_list = [int(re.search('images_identity/(.*).png', f).group(1)) for f in file_dir_list]
file_dir_list = file_dir_list[np.argsort(file_list)]

batch_size = 6
num_image = 10
x_seq_image = np.zeros((num_image, 3, 40, 120))
x_seq_label = [0, 2, 1, 2, 0, 0, 1, 2, 2, 0]
for ith_ima in list(range(0, num_image)):
    x_seq_image[ith_ima, ...] = image.imread(file_dir_list[x_seq_label[ith_ima]]).reshape(3, 40, 120)

y_seq_label = x_seq_label[1:]
y_seq_label.append(x_seq_label[0])
y_seq_label = np.array(y_seq_label)

y_seq_image = np.zeros((num_image, 3, 40, 120))
for ith_ima in list(range(0, num_image)):
    y_seq_image[ith_ima, ...] = image.imread(file_dir_list[y_seq_label[ith_ima]]).reshape(3, 40, 120)

tensor_image_x = pt.Tensor(x_seq_image).type(pt.float)
tensor_image_y = pt.Tensor(y_seq_image).type(pt.float)
tensor_label_y = pt.Tensor(y_seq_label).type(pt.LongTensor)
custom_dataset = TensorDataset(tensor_image_x, tensor_image_y)
my_dataloader = DataLoader(custom_dataset, batch_size=batch_size, drop_last=True)




output_array = np.zeros((2, 2))
ks_pool = [5, 20]
abs_pool = [25, 50]

for ith_ks in list(range(0, 2)):
    for ith_abs in list(range(0, 2)):
        kernel_size = ks_pool[ith_ks]
        num_abstract = abs_pool[ith_abs]
        if kernel_size == 5:
            dim_input = 116 * 36
            dim_input_x = 116
            dim_input_y = 36
        if kernel_size == 20:
            dim_input = 101 * 21
            dim_input_x = 101
            dim_input_y = 21

        class ConV_lstm_autoencoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 20, kernel_size)
                self.fc1 = nn.Linear(dim_input * 20, num_abstract)
                self.lstm = nn.LSTM(input_size=num_abstract, hidden_size=num_abstract, num_layers=1, batch_first=True)
                self.fc2 = nn.Linear(num_abstract, dim_input * 20)
                self.t_conv = nn.ConvTranspose2d(20, 3, kernel_size)
            def forward(self, x):
                x = F.relu(self.conv(x))
                x = pt.flatten(x, 1)
                x = F.relu(self.fc1(x))
                x, _ = self.lstm(x)
                x = F.relu(self.fc2(x))
                x = x.reshape(batch_size, 20, dim_input_y, dim_input_x)
                x = F.sigmoid(self.t_conv(x))
                return x
        conv_lstm_weight = pt.load('./weight_conv_lstm_deconv_ks' + str(kernel_size) + '_numAbs' + str(num_abstract) + '.pth')
        my_net = ConV_lstm_autoencoder()

        my_net.load_state_dict(conv_lstm_weight)
        counter = 0
        num_corr = 0
        with pt.no_grad():
            # x seq [0, 2, 1, 2, 0, 0, 1, 2, 2, 0]
            # y seq [2, 1, 2, 0, 0, 1, 2, 2, 0, 0]
            for ith_attempt in list(range(0, 10)):
                ref_index = random.choice(list(range(0, 3)))
                cur_x_segment = []
                cur_y_segment = []
                for ith in list(range(0, batch_size)):
                    cur_x_segment.append(x_seq_label[ref_index + ith])
                    cur_y_segment.append(y_seq_label[ref_index + ith])
                input_ima = np.zeros((batch_size, 3, 40, 120))
                for ith in list(range(0, len(cur_x_segment))):
                    input_ima[ith, ...] = image.imread(file_dir_list[cur_x_segment[ith]]).reshape(3, 40, 120)
                outputs = my_net(pt.Tensor(input_ima).type(pt.float))
                outputs = outputs.reshape(batch_size, 40, 120, 3).detach().numpy()

                for ith in list(range(0, batch_size)):
                    cur_pred = np.mean(outputs[ith], axis=2)
                    index_pred = np.argmin(
                        [np.mean(cur_pred[:, 0:40]), np.mean(cur_pred[:, 41:80]), np.mean(cur_pred[:, 81:120])])
                    if index_pred == cur_y_segment[ith]:
                        num_corr = num_corr + 1
                    counter = counter + 1
        print(str(counter) + '-' + str(num_corr))
        output_array[ith_ks, ith_abs] = num_corr/counter


# plot bar
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.close('all')
font = {'family': 'Arial',
        'weight': 'normal',
        'size': 70}
matplotlib.rc('font', **font)
fig = plt.figure(figsize=(13, 15))
ax = fig.add_subplot()
bars = ax.bar([1.5, 3.5, 5.5, 7.5], [output_array[0,0], output_array[0,1], output_array[1,0], output_array[1,1]], tick_label=['1', '2', '3', '4'], width=0.7, edgecolor='black', linewidth=3)
bars[0].set_color('orangered')
bars[1].set_color('salmon')
bars[2].set_color('thistle')
bars[3].set_color('blueviolet')
for axis in ['top','right']:
    ax.spines[axis].set_linewidth(0)
for axis in ['bottom','left']:
    ax.spines[axis].set_linewidth(5)
plt.xticks([0.5, 2.5, 4.5, 6.5], labels=['ks=5, as=25','ks=5, as=50','ks=20, as=25','ks=20, as=50'],rotation=45)
plt.yticks([0, 0.3, 1])
plt.ylabel('Correct rate')
plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.4)
plt.show()




# plt.close('all')
# fig = plt.figure(figsize=(7, 8))
# ax = fig.add_subplot()
# ax.imshow(outputs[2].reshape(40, 120, 3))
# plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
# ax.get_xaxis().set_ticks([])
# ax.get_yaxis().set_ticks([])
# plt.show()
#
# plt.close('all')
# fig = plt.figure(figsize=(7, 8))
# ax = fig.add_subplot()
# ax.imshow(y_seq_image[3].reshape(40, 120, 3))
# plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
# ax.get_xaxis().set_ticks([])
# ax.get_yaxis().set_ticks([])
# plt.show()


# plt.close('all')
# fig = plt.figure(figsize=(7, 3))
# ax = fig.add_subplot()
# ax.imshow(input_ima[ith].reshape(40, 120, 3))
# plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
# ax.get_xaxis().set_ticks([])
# ax.get_yaxis().set_ticks([])
# for axis in ['top', 'right', 'bottom', 'left']:
#     ax.spines[axis].set_linewidth(5)
# plt.savefig('/Users/bo/Documents/PycharmProjects/dingwei_relational_memory/temp_test/stimuli_' + str(counter) + '.png')
#
# plt.close('all')
# fig = plt.figure(figsize=(7, 3))
# ax = fig.add_subplot()
# ax.imshow(corr_ima[ith].reshape(40, 120, 3))
# plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
# ax.get_xaxis().set_ticks([])
# ax.get_yaxis().set_ticks([])
# for axis in ['top', 'right', 'bottom', 'left']:
#     ax.spines[axis].set_linewidth(5)
# plt.savefig('/Users/bo/Documents/PycharmProjects/dingwei_relational_memory/temp_test/next_' + str(counter) + '.png')
#
# plt.close('all')
# fig = plt.figure(figsize=(7, 3))
# ax = fig.add_subplot()
# ax.imshow(outputs[ith].reshape(40, 120, 3))
# if index_pred == cur_y_segment[ith]:
#     rect = patches.Rectangle((index_pred*40+20-11, 10), 20, 20, linewidth=8, edgecolor='green', facecolor='none')
#     ax.add_patch(rect)
# else:
#     rect = patches.Rectangle((index_pred*40+20-11, 10), 20, 20, linewidth=8, edgecolor='red', facecolor='none')
#     ax.add_patch(rect)
# plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
# ax.get_xaxis().set_ticks([])
# ax.get_yaxis().set_ticks([])
# for axis in ['top', 'right', 'bottom', 'left']:
#     ax.spines[axis].set_linewidth(5)
# plt.savefig('/Users/bo/Documents/PycharmProjects/dingwei_relational_memory/temp_test/pred_' + str(counter) + '.png')










import numpy as np
import matplotlib.pyplot as plt
import re
import moviepy.video.io.ImageSequenceClip
from os import listdir
import imageio as iio
from matplotlib import gridspec
import matplotlib
mypath = '/Users/bo/Documents/PycharmProjects/dingwei_relational_memory/weights/'
pred_list = np.array([mypath + f for f in listdir(mypath) if "pred" in f])
corr_index_list = [int(re.search('pred_(.*).png', f).group(1)) for f in pred_list]
pred_list = pred_list[np.argsort(corr_index_list)]

next_list = np.array([mypath + f for f in listdir(mypath) if "next_" in f])
corr_index_list = [int(re.search('next_(.*).png', f).group(1)) for f in next_list]
next_list = next_list[np.argsort(corr_index_list)]

stimu_list = np.array([mypath + f for f in listdir(mypath) if "stimuli_" in f])
corr_index_list = [int(re.search('stimuli_(.*).png', f).group(1)) for f in stimu_list]
stimu_list = stimu_list[np.argsort(corr_index_list)]

for ith in list(range(0, len(pred_list))):
    gap = 0.5
    plt.close('all')
    font = {'family': 'Arial',
            'weight': 'normal',
            'size': 70}
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(11, 15))
    gs = gridspec.GridSpec(3, 1, wspace=gap, hspace=gap)
    gs.update(left=0, right=1, top=0.9, bottom=0, wspace=gap, hspace=gap)
    img_pred = iio.v3.imread(pred_list[ith])
    img_next = iio.v3.imread(next_list[ith])
    img_stimu = iio.v3.imread(stimu_list[ith])
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(img_stimu)
    ax1.set_title('Stimulus ' + str(ith))
    ax1.axis('off')
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(img_next)
    ax2.set_title('Next stimulus')
    ax2.axis('off')
    ax3 = fig.add_subplot(gs[2])
    ax3.imshow(img_pred)
    ax3.set_title('predicted stimulus')
    ax3.axis('off')
    plt.savefig('/Users/bo/Documents/PycharmProjects/dingwei_relational_memory/temp_test/animation_' + str(ith) + '.png')


mypath = '/Users/bo/Documents/PycharmProjects/dingwei_relational_memory/temp_test/'
anaim_list = np.array([mypath + f for f in listdir(mypath) if "animation_" in f])
corr_index_list = [int(re.search('animation_(.*).png', f).group(1)) for f in anaim_list]
anaim_list = anaim_list[np.argsort(corr_index_list)]

fps = 2
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(anaim_list.tolist(), fps=fps)
clip.write_videofile('/Users/bo/Desktop/convolutional_lstm_performance.mp4')





