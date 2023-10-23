import copy
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from sklearn.metrics import f1_score
import openpyxl
import numpy as np
import xlwt
import re
from utils.utils import get_top_grad_index

label_dict={'browsing': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'Email': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            'facebook': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            'FILE': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            'MAIL': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            'Skype': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            'spotify': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            'tor': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            'VOIP': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            'Youtube': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}

def solve_input(image): #处理输入归一化,输入是(32,32)的未归一化的numpy图片，输出是(1,32,32)的tensor
    image = image.astype(np.float32)
    image = image / 255
    image = torch.tensor(image)
    image = image.reshape((1,32,32))
    image.requires_grad = True
    return image


def I_FGSM(model,x,labels,alpha=8/255,iteration=100):   #labels int
    ori_image = x.data
    loss = nn.CrossEntropyLoss()
    for i in range(0,iteration):
        x.requires_grad = True
        outputs = model(x)
        if(outputs.argmax(axis=1).item()!=labels):
            break
        model.zero_grad()
        tensor_label = torch.tensor(labels).unsqueeze(0)
        cost = loss(outputs,tensor_label)
        cost.backward()
        x = x+alpha*x.sign()
        x = torch.clamp(x, min=0, max=1).detach_()
    a = torch.nonzero(ori_image-x)
    return x,len(a)

def re_I_FGSM(model,x,labels,num=15,eps=0.3,alpha=8/255,iteration=100):
    ori_image = x.data
    loss = nn.CrossEntropyLoss()
    for i in range(0,iteration):
        x.requires_grad = True
        outputs = model(x)
        if(outputs.argmax(axis=1).item()!=labels):
            break
        model.zero_grad()
        tensor_label = torch.tensor(labels).unsqueeze(0)
        cost = loss(outputs,tensor_label)
        cost.backward()
        # 获取num点
        index_list = get_top_grad_index(x.grad.data, num, 2)
        data_grad = torch.zeros(x.grad.data.shape)
        for i in range(len(index_list)):
            data_grad[index_list[i][0],index_list[i][1],index_list[i][2]] = x.grad.data[index_list[i][0],index_list[i][1],index_list[i][2]]
        x = x + alpha*data_grad.sign()
        x = torch.clamp(x, min=0, max=1).detach_()
        # eta = torch.clamp(x - ori_image, min=-eps, max=eps)
        # x = torch.clamp(ori_image+eta, min=0, max=1).detach_()
    a = torch.nonzero(ori_image-x)
    return x,len(a)



def I_FGSM_attack(raw_path):
    model = torch.load("../train_model/model.pkl")
    label_count = {}  # 统计每个预测无错的label数量
    SR_dict = {}
    AP_dict = {}
    raw_label = []
    new_label=[]
    count = 0
    patten = '[a-zA-Z]+'

    for file in tqdm(os.listdir(raw_path)):  # 对一张图片开始实验
        label = re.match(patten, file).group()
        int_label = torch.tensor(label_dict[label]).argmax().item()
        raw_label.append(int_label)
        image_path = raw_path + '/' + file
        # load image
        raw_image = np.load(image_path)
        image = copy.deepcopy(raw_image)
        # preprocess input
        image = solve_input(image)
        predict = model(image)
        if (predict.argmax() == int_label):
            if (label in label_count):
                label_count[label] = label_count[label] + 1
            else:
                label_count[label] = 1
            pertubaed_image,point = I_FGSM(model, image, int_label)
            new_predict = model(pertubaed_image)
            new_label.append(int(new_predict.argmax().item()))

            if (new_predict.argmax() != int_label):
                if (label in AP_dict):
                    AP_dict[label] = AP_dict[label] + point
                else:
                    AP_dict[label] = point
                if (label in SR_dict):
                    SR_dict[label] = SR_dict[label] + 1
                else:
                    SR_dict[label] = 1
        else:
            new_label.append(int(predict.argmax().item()))
        count = count+1
    SR_all = 0
    AP_all = 0
    for key in label_count:
        AP_dict[key] = AP_dict[key] / SR_dict[key]
        SR_dict[key] = SR_dict[key] / label_count[key]
        SR_all = SR_all + SR_dict[key]
        AP_all = AP_all + AP_dict[key]
        print("label", key, "count", label_count[key], "SR", SR_dict[key], "AP", AP_dict[key], "F1-score(micro)",
              f1_score(raw_label, new_label, average='micro'),"F1-score(macro)",
              f1_score(raw_label, new_label, average='macro'))
    print("SR_all:",SR_all/10,"AP_all:",AP_all/10)
    return SR_all/10,AP_all/10,f1_score(raw_label, new_label, average='macro'),f1_score(raw_label, new_label, average='micro')




def RE_I_FGSM_attack(raw_path,num):
    model = torch.load("../train_model/model.pkl")
    label_count = {}  # 统计每个预测无错的label数量
    SR_dict = {}
    AP_dict = {}
    raw_label = []
    new_label=[]
    count = 0
    patten = '[a-zA-Z]+'

    for file in tqdm(os.listdir(raw_path)):  # 对一张图片开始实验
        label = re.match(patten, file).group()
        int_label = torch.tensor(label_dict[label]).argmax().item()
        raw_label.append(int_label)
        image_path = raw_path + '/' + file
        # load image
        raw_image = np.load(image_path)
        image = copy.deepcopy(raw_image)
        # preprocess input
        image = solve_input(image)
        predict = model(image)
        if (predict.argmax() == int_label):
            if (label in label_count):
                label_count[label] = label_count[label] + 1
            else:
                label_count[label] = 1
            pertubaed_image,point = re_I_FGSM(model, image, int_label, num)
            new_predict = model(pertubaed_image)
            new_label.append(int(new_predict.argmax().item()))

            if (new_predict.argmax() != int_label):
                if (label in AP_dict):
                    AP_dict[label] = AP_dict[label] + point
                else:
                    AP_dict[label] = point
                if (label in SR_dict):
                    SR_dict[label] = SR_dict[label] + 1
                else:
                    SR_dict[label] = 1
        else:
            new_label.append(int(predict.argmax().item()))
        count = count+1
    SR_all = 0
    AP_all = 0
    for key in label_count:
        AP_dict[key] = AP_dict[key] / SR_dict[key]
        SR_dict[key] = SR_dict[key] / label_count[key]
        SR_all = SR_all + SR_dict[key]
        AP_all = AP_all + AP_dict[key]
        print("label", key, "count", label_count[key], "SR", SR_dict[key], "AP", AP_dict[key], "F1-score(micro)",
              f1_score(raw_label, new_label, average='micro'),"F1-score(macro)",
              f1_score(raw_label, new_label, average='macro'))
    print("SR_all:",SR_all/10,"AP_all:",AP_all/10)
    return num,SR_all/10,AP_all/10,f1_score(raw_label, new_label, average='macro'),f1_score(raw_label, new_label, average='micro')




def statistic_fields(raw_path,num=15):
    print("------------------statistic_fields-----------------")
    model = torch.load('../train_model/model.pkl')
    label_result={}
    patten = '[a-zA-Z]+'
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    for file in tqdm(os.listdir(raw_path)):  # 对一张图片开始实验
        label = re.match(patten, file).group()
        int_label = torch.tensor(label_dict[label]).argmax().item()
        image_path = raw_path + '/' + file
        # load image
        raw_image = np.load(image_path)
        image = copy.deepcopy(raw_image)
        # preprocess input
        image = solve_input(image)
        predict = model(image)
        if(label not in label_result):
            label_result[label]={}
        if (predict.argmax() == int_label):
            pertubaed_image,point = re_I_FGSM(model, image, int_label, num)
            a = torch.nonzero(pertubaed_image-image)
            for i in range(0,point):
                temp = tuple(a[i].numpy()[1:])
                if(temp in label_result[label]):
                    label_result[label][temp] = label_result[label][temp]+1
                else:
                    label_result[label][temp] = 1

    for i in label_dict:
        sheet = book.add_sheet(i, cell_overwrite_ok=True)
        sorted_item = sorted(label_result[i].items(),key=lambda x:x[1],reverse=True)
        for j in range(0,10):
            sheet.write(j, 0, str(sorted_item[j][0]))
            sheet.write(j, 1, str(sorted_item[j][1]))

    book.save("I-FGSM_field.xls")
    return None


import time
from utils.recode import recode_data
if __name__ == "__main__":
    # #统计不同攻击点数的结果记录
    raw_path = "../dataset/npy_dataset/test"
    # col = ['k', 'SR', 'AP', 'score(micro)', 'score(macro)','time_cost']
    # for i in range(5,16):
    #     start = time.time()
    #     rows = RE_I_FGSM_attack(raw_path, i)
    #     rows = list(rows)
    #     end_time = time.time()
    #     rows.append(end_time-start)
    #     recode_data("I_FGSM",rows,col)

    # #加了限制，每次限制15点的攻击效果
    # start = time.time()
    # RE_I_FGSM_attack(raw_path,15)
    # end_time = time.time()
    # print("time cost:",end_time-start)

    #加了限制，每次限制15点的攻击效果
    start = time.time()
    statistic_fields(raw_path,15)
    end_time = time.time()
    print("time cost:",end_time-start)