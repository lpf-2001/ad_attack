import numpy as np
import torch
from sklearn.metrics import f1_score
import copy
import os
import re
labeldict={}
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

def test_accuracy(model,x,target):
    predict = model(x)
    accuracy = ((predict.argmax(axis=1)==target.argmax(axis=1)).float().sum().item())/target.shape[0]
    macro_f1 = f1_score(target.argmax(axis=1), predict.argmax(axis=1), average='macro')
    micro_f1 = f1_score(target.argmax(axis=1), predict.argmax(axis=1), average='micro')
    print("macro_f1:",macro_f1,"micro_f1:",micro_f1)
    return accuracy



def test_data_accuracy(model):
    test_datapath = "D:\\一无所获的大学生活\项目组\\AD_attack2\\dataset\\npy_dataset\\test"
    x_test, y_test = deal_image(test_datapath)
    x_test = x_test.astype(np.float32)
    x_test = torch.tensor(x_test)
    # x_test = x_test.reshape(1,32,32)
    y_test = torch.tensor(y_test)
    y_test = y_test.argmax(axis=1)



    accuracy = test_accuracy(model,x_test,y_test)
    print("test_data accuracy is:",accuracy)
    return accuracy