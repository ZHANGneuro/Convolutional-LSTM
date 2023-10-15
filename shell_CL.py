import numpy as np
from os import listdir
import re
from matplotlib import image
import torch as pt
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

kernel_size = int(sys.argv[1])
num_abstract = int(sys.argv[2])
ith_a = int(sys.argv[3])

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

class ConV_lstm_autoencoder(nn.Module):
    def __init__(self, kernel_size, dim_input, num_abstract):
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

dim_input_x = image_width - kernel_size + 1
dim_input_y = image_height - kernel_size + 1
dim_input = dim_input_x * dim_input_y
my_net = ConV_lstm_autoencoder(kernel_size, dim_input, num_abstract)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(my_net.parameters(), lr=1e-3, momentum=0.9)

for epoch in range(1000):
    print(str(kernel_size) + '-' + str(num_abstract) + '-' + str(epoch))
    for i, data in enumerate(my_dataloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = my_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

PATH = './weights/weight_conv_lstm_ks' + str(kernel_size) + '_numAbs' + str(num_abstract) + '_ith' + str(ith_a) + '.pth'
pt.save(my_net.state_dict(), PATH)
