
import numpy as np
from os import listdir
import re
from matplotlib import image
import torch as pt
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

kernel_size = int(sys.argv[1])
num_abstract = int(sys.argv[2])
a_attempt = int(sys.argv[3])

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

conv_lstm_weight = pt.load('./weights/weight_conv_lstm_ks' + str(kernel_size) + '_numAbs' + str(num_abstract) + '_ith' + str(a_attempt) + '.pth')
my_net = ConV_lstm_autoencoder()
with pt.no_grad():
    my_net.conv.weight = nn.Parameter(conv_lstm_weight['conv.weight'], requires_grad=False)
    my_net.conv.bias = nn.Parameter(conv_lstm_weight['conv.bias'], requires_grad=False)
    my_net.fc1.weight = nn.Parameter(conv_lstm_weight['fc1.weight'], requires_grad=False)
    my_net.fc1.bias = nn.Parameter(conv_lstm_weight['fc1.bias'], requires_grad=False)
    my_net.lstm.weight_ih_l0 = nn.Parameter(conv_lstm_weight['lstm.weight_ih_l0'], requires_grad=False)
    my_net.lstm.weight_hh_l0 = nn.Parameter(conv_lstm_weight['lstm.weight_hh_l0'], requires_grad=False)
    my_net.lstm.bias_ih_l0 = nn.Parameter(conv_lstm_weight['lstm.bias_ih_l0'], requires_grad=False)
    my_net.lstm.bias_hh_l0 = nn.Parameter(conv_lstm_weight['lstm.bias_hh_l0'], requires_grad=False)

criterion = nn.MSELoss()
optimizer = optim.Adam(my_net.parameters(), lr=0.001, weight_decay=1e-5)
for epoch in range(1000):
    for i, data in enumerate(my_dataloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = my_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(str(epoch) + '-' + str(loss.item()))
PATH = './weights/weight_conv_lstm_deconv_ks' + str(kernel_size) + '_numAbs' + str(num_abstract) + '_a' + str(a_attempt) + '.pth'
pt.save(my_net.state_dict(), PATH)
