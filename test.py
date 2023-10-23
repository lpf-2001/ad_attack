import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import copy

def get_index(grad,value):
    for i in range(grad.shape[0]):
        for j in range(grad.shape[1]):
            for k in range(grad.shape[2]):
                if(grad[i,j,k]==value):
                    return np.unravel_index(i*grad.shape[1]+j*grad.shape[2]+k,grad.shape)
    return None

def get_top_grad_index(grad,num,flag=1):   #返回梯度最大的num个点
    copy_grad = copy.deepcopy(grad)
    copy_grad = torch.abs(copy_grad)         #这一行待定
    index_list = []
    if(flag==1):
        count = 0
        for i in range(0, 32):
            for j in range(0, 32):
                count = count + 1

                copy_grad[0, i, j] = -1
                if (count > 34):
                    break
            if (count >34):
                break
        # 一代版本
        # print(type(copy_grad))
        for i in range(num):
            value = torch.max(copy_grad)
            index = get_index(copy_grad, value)
            if (index is None):
                assert "one fault"
            index_list.append(index)
            copy_grad[index[0], index[1], index[2]] = -1

    elif(flag==2):
        copy_grad = copy_grad.reshape((1, 1024))
        for i in range(0, 34):
            copy_grad[0, i] = -1
        sort_vector = copy_grad.sort(1, True)[0]
        choose_num = sort_vector[0, num]
        compare_metrix = torch.ones_like(copy_grad) * choose_num
        compare_index = copy_grad.gt(compare_metrix)
        compare_index = compare_index.reshape((1, 32, 32))
        for i in range(0, 32):
            for j in range(0, 32):
                if (compare_index[0, i, j]):
                    index_list.append((0, i, j))

    return index_list


def solve_input(image): #处理输入归一化,输入是(32,32)的未归一化的numpy图片，输出是(1,32,32)的tensor
    image = image.astype(np.float32)
    image = image / 255
    image = torch.tensor(image)
    image = image.reshape((1,32,32))
    image.requires_grad = True
    return image
loss = nn.CrossEntropyLoss()
data = np.load("dataset/npy_dataset/test/facebook_Audio_test8.npy")

model = torch.load('./train_model/model.pkl')
data = solve_input(data)
data = torch.reshape(data,(32,32))
data = data.detach().numpy()
plt.imshow(data)
plt.show()
output = model(data)
model.zero_grad()
cost = loss(output, torch.unsqueeze(torch.tensor(2), 0))
cost.backward()
grad = data.grad.data


K = 15
index_list = get_top_grad_index(grad, K,2)
data_grad = torch.zeros(grad.shape)
for i in range(len(index_list)):
    data_grad[index_list[i][0], index_list[i][1], index_list[i][2]] = grad[
        index_list[i][0], index_list[i][1], index_list[i][2]]
data_grad = torch.reshape(data_grad,(32,32))
plt.imshow(data_grad)
plt.show()
print(output)