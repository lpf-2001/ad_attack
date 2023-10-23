import copy
import os
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import torch.nn as nn
import re
import torch

import numpy as np
from  utils.utils import get_top_grad_index
from tqdm import tqdm
import xlwt

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

def pgd_attack(model, images, labels, eps=0.3, alpha=8/255, iters=100) :

    loss = nn.CrossEntropyLoss()
    # 原图像
    ori_images = images.data

    for i in range(iters) :
        images.requires_grad = True
        outputs = model(images)
        if(outputs.argmax(axis=1).item()!=labels):
            break
        model.zero_grad()
        cost = loss(outputs, torch.unsqueeze(torch.tensor(labels),0))
        cost.backward()
        # 图像 + 梯度得到对抗样本
        adv_images = images + alpha*images.grad.sign()
        # 限制扰动范围
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        # 进行下一轮对抗样本的生成。破坏之前的计算图
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
    a = torch.nonzero(ori_images-images)

    return outputs.argmax(axis=1).item(),len(a)

def re_pgd_attack(model, images, labels, num, eps=0.3, alpha=8/255, iters=100):
    loss = nn.CrossEntropyLoss()
    # 原图像
    ori_images = images.data
    for i in range(iters) :
        if(i==0):
            #初始化随机扰动
            eta = torch.zeros_like(images).uniform_(-eps, eps)
            images.requires_grad = True
            outputs = model(images)
            model.zero_grad()
            cost = loss(outputs, torch.unsqueeze(torch.tensor(labels),0))
            cost.backward()
            # 图像 + 梯度得到对抗样本
            index_list = get_top_grad_index(images.grad.data, num,2)
            new_eta = torch.zeros_like(eta)
            #只改动筛选的点
            for j in range(len(index_list)):
                new_eta[index_list[j][0],index_list[j][1],index_list[j][2]] = eta[index_list[j][0],index_list[j][1],index_list[j][2]]
            eta = new_eta
            eta = torch.clamp(eta, min=-eps, max=eps)
        else:
            images.requires_grad = True
            outputs = model(images+eta)

            model.zero_grad()
            cost = loss(outputs, torch.unsqueeze(torch.tensor(labels),0))
            cost.backward()
            # 图像 + 梯度得到对抗样本

            index_list = get_top_grad_index(images.grad.data, num,2)
            data_grad = torch.zeros(images.grad.data.shape)

            for i in range(len(index_list)):
                data_grad[index_list[i][0],index_list[i][1],index_list[i][2]] = images.grad.data[index_list[i][0],index_list[i][1],index_list[i][2]]
            adv_images = images + alpha * data_grad.sign()
            # 限制扰动范围
            eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)



        # 进行下一轮对抗样本的生成。破坏之前的计算图
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
        new_predict = model(images)
        if(new_predict.argmax(axis=1).item()!=labels):
            break
    if(new_predict.argmax()!=labels):
        count_df = len(torch.nonzero(ori_images-images))
    else:
        count_df = 0

    return images,count_df



def PGD_test(raw_path):
    print("-------------------PGD_test-----------------")
    model = torch.load('../train_model/model.pkl')
    patten = '[a-zA-Z]+'
    label_count = {}  # 统计每个预测无错的label数量
    SR_dict = {}
    AP_dict = {}
    raw_label = []
    new_label = []
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
            new_predict, point = pgd_attack(model, image, int_label)
            new_label.append(new_predict)
            if (new_predict != int_label):
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

    SR_all = 0
    AP_all = 0
    for key in label_count:
        AP_dict[key] = AP_dict[key] / SR_dict[key]
        SR_dict[key] = SR_dict[key] / label_count[key]
        SR_all = SR_all + SR_dict[key]
        AP_all = AP_all + AP_dict[key]
        print("label", key, "count", label_count[key], "SR", SR_dict[key], "AP", AP_dict[key], "F1-score(micro)",
              f1_score(raw_label, new_label, average='micro'), "F1-score(macro)",
              f1_score(raw_label, new_label, average='macro'))
    print("SR_all:", SR_all / 10, "AP_all:", AP_all / 10)
    return SR_all / 10, AP_all / 10, f1_score(raw_label, new_label, average='micro'), f1_score(raw_label, new_label,
                                                                                               average='macro')


def RE_PGD_test(raw_path,num=15):
    print("------------------RE_PGD_test-----------------")
    model = torch.load('../train_model/model.pkl')
    patten = '[a-zA-Z]+'
    label_count = {}  # 统计每个预测无错的label数量
    SR_dict = {}
    AP_dict = {}
    raw_label = []
    new_label=[]
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
            pertubted_image, point = re_pgd_attack(model, image, int_label, num)
            new_predict = model(pertubted_image).argmax(axis=1).item()
            new_label.append(new_predict)

            if (new_predict != int_label):
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
    return num,SR_all/10,AP_all/10,f1_score(raw_label, new_label, average='micro'),f1_score(raw_label, new_label, average='macro')

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
            pertubaed_image,point = re_pgd_attack(model, image, int_label, num)
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

    book.save("PGD_field.xls")
    return None


import time
from utils.recode import recode_data

if __name__ == "__main__":
    # raw_path = "../dataset/npy_dataset/test"
    # #原来方法不加限制攻击效果
    # # PGD_test(raw_path)
    # #统计单纯topk方法攻击15点实验结果
    # start = time.time()
    # RE_PGD_test(raw_path,15)
    # end_time = time.time()
    # print("time cost:",end_time-start)

    # #统计不同攻击点数的结果记录
    # raw_path = "../dataset/npy_dataset/test"
    # col = ['k', 'SR', 'AP', 'score(micro)', 'score(macro)','time_cost']
    # for i in range(5,16):
    #     start = time.time()
    #     rows = RE_PGD_test(raw_path, i)
    #     rows = list(rows)
    #     end_time = time.time()
    #     rows.append(end_time-start)
    #     recode_data("PGD",rows,col)

    raw_path = "../dataset/npy_dataset/test"
    start = time.time()
    statistic_fields(raw_path,15)
    end_time = time.time()
    print("time cost:",end_time-start)