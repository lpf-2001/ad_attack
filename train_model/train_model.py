import os
from model import Net
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from evaluate import test_accuracy
from utils.evaluate import evaluate_train,solve_input
import copy
import re
labeldict = {}
def deal_image(path, classname=None):
    count = 0
    templabels = [0 for i in range(10)]
    images = []
    labels = []
    if classname == None:
        imagenamelist = [path + "\\" + name for name in os.listdir(path) if name.lower().endswith('npy')]#找到所有图片路径
    else:
        imagenamelist = [path + "\\" + name for name in os.listdir(path) if
                         name.lower().endswith('npy') and name.lower().startswith(classname)]

    for i in imagenamelist:
        image = np.load(i)
        image = image[np.newaxis,:, :]
        images.append(image)
        pattern = re.compile('^[a-z]+')
        vpnpattern = re.compile('(vpn_[a-z]+)')
        name = i.split('\\')[-1]
        if name.startswith('vpn'):
            name = vpnpattern.findall(name.lower())[0]
        else:
            name = pattern.findall(name.lower())[0]    #get label
        if name in labeldict:
            label = labeldict[name]
            labels.append(label)
            count += 1
        else:
            labellength = len(labeldict)
            templabel = copy.deepcopy(templabels)

            templabel[labellength] = 1
            labeldict.update({name: templabel})
            label = templabel
            labels.append(label)
            count += 1
    images = np.array(images)
    images = images / 255.0
    labels = np.array(labels)
    with open('label.txt','a+') as f:
        f.write("\nlabel dict:")
        f.write(str(labeldict))
    return images, labels




def train_model(x,y,batch_size=256,epoch=15):
    x = x.astype(np.float32)
    x = torch.tensor(x)
    y = torch.tensor(y)
    dataset = Data.TensorDataset(x, y)
    data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
    model = Net(batch_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    cross_loss = nn.CrossEntropyLoss()
    for i in range(epoch):
        count = 0
        for X, Y in data_iter:
            count = count+1
            predict = model(X)
            Y = Y.argmax(axis=1)
            loss = cross_loss(predict, Y)
            print("epoch:",i," batch:",count," accuracy is:",test_accuracy(model, X, Y))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

batch_size = 256
epoch = 20
datapath = "D:\\一无所获的大学生活\项目组\\AD_attack2\\dataset\\npy_dataset\\test"

if __name__ == "__main__":

    if(os.path.exists('model.pkl')):

        model = torch.load('model.pkl')
    else:
        x, y = deal_image(datapath)
        model = train_model(x,y,batch_size,epoch)
    evaluate_train(model,raw_path=datapath)
    # test_data_accuracy(model)
    # if(~os.path.exists('model.pkl')):
    #
    # torch.save(model,"model.pkl")