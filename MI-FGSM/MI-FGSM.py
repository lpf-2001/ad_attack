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

def solve_input(image): #处理输入归一化,输入是(32,32)的未归一化的numpy图片，输出是(1,32,32)的tensor
    image = image.astype(np.float32)
    image = image / 255
    image = torch.tensor(image)
    image = image.reshape((1,32,32))
    image.requires_grad = True
    return image

field_dict = {
    "dst_mac":0,#0-5
    "src_mac":0,#5-11
    "Type":0,#11-13
    "VHL":0,#ip:version and header length 14
    "ECN":0,#15
    "Total_length":0,#16-17
    "Identification":0,#18-19
    "Flags":0,#20-21
    "TTL":0, #22
    "Protocol":0,#23
    "Header_Checksum":0,#24-25
    "src_ip":0,#26-29
    "dst_ip":0, #30-33
    "Sport":0, #34-35  传输层
    "Dport":0,#36-37
    "Snumber":0,#38-41
    "ACK_number":0,#42-45
    "flags":0,#46-47
    "Windows":0,#48-49
    "Checksum":0,#50-51
    "Urgent_pointer":0,#52-53

    "Options":0,#54-65 Timestamp echo reply
    "others":0
}
def cast_pcap(s):   #传入一个点s 元组类型
    loc = s[0]*32+s[1]
    if(0<=loc and loc<=5):
        #mac dst address
        field_dict["dst_mac"] = field_dict["dst_mac"]+1
    elif(loc<=11):
        field_dict["src_mac"] = field_dict["src_mac"]+1
    elif(loc<=13):
        field_dict["Type"] = field_dict["Type"] + 1
    elif(loc<=14):
        field_dict["VHL"] = field_dict["VHL"] + 1
    elif(loc<=15):
        field_dict["ECN"] = field_dict["ECN"] + 1
    elif(loc<=17):
        field_dict["Total_length"] = field_dict["Total_length"] + 1
    elif(loc<=19):
        field_dict["Identification"] = field_dict["Identification"] + 1
    elif(loc<=21):
        field_dict["Flags"] = field_dict["Flags"] + 1
    elif(loc<=22):
        field_dict["TTL"] = field_dict["TTL"] + 1
    elif(loc<=23):
        field_dict["Protocol"] = field_dict["Protocol"]+1
    elif(loc<=25):
        field_dict["Header_Checksum"] = field_dict["Header_Checksum"] + 1
    elif(loc<=29):
        field_dict["src_ip"] = field_dict["src_ip"] + 1
    elif(loc<=33):
        field_dict["dst_ip"] = field_dict["dst_ip"] + 1
    elif(loc<=35):
        field_dict["Sport"] = field_dict["Sport"] + 1
    elif(loc<=37):
        field_dict["Dport"] = field_dict["Dport"] + 1
    elif(loc<=41):
        field_dict["Snumber"] = field_dict["Snumber"] + 1
    elif(loc<=45):
        field_dict["ACK_number"] = field_dict["ACK_number"] + 1
    elif(loc<=47):
        field_dict["Flags"] = field_dict["Flags"] + 1
    elif(loc<=49):
        field_dict["Windows"] = field_dict["Windows"] + 1
    elif(loc<=51):
        field_dict["Checksum"] = field_dict["Checksum"] + 1
    elif(loc<=53):
        field_dict["Urgent_pointer"] = field_dict["Urgent_pointer"] + 1
    elif(loc<=65):
        field_dict["Options"] = field_dict["Options"] + 1
    else:
        field_dict["others"] = field_dict["others"]+1
    # print(field_dict)

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




def MI_FGSM_attack(model, images, labels, eps=0.3, alpha=8/255, iters=100, decay=1) :

    loss = nn.CrossEntropyLoss()
    # 原图像
    ori_images = images.data
    # print(ori_images)
    momentum = torch.zeros_like(images)
    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)
        if (outputs.argmax(axis=1) != labels):
            for i in range(0, 32):
                for j in range(0, 32):
                    if (ori_images[0, i, j] != images[0, i, j]):
                        count_df = count_df + 1
            return images, count_df + 1
        model.zero_grad()
        cost = loss(outputs, torch.unsqueeze(torch.tensor(labels),0))
        cost.backward()
        #动量项更新
        momentum = decay * momentum + images.grad.sign()
        # 图像 + 梯度得到对抗样本
        adv_images = images + alpha*momentum.sign()
        # 限制扰动范围
        # eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        eta = adv_images-ori_images
        # 进行下一轮对抗样本的生成。破坏之前的计算图
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
        # images = (ori_images+eta).detach()
        count_df = 0
    # print(images)
    for i in range(0, 32):
        for j in range(0, 32):
            if (ori_images[0, i, j] != images[0, i, j]):
                count_df = count_df + 1
    return images,count_df


point_dict = {}

#num:一次攻击的点数 alpha：攻击强度 iters：迭代次数 decay：动量项衰减零
def re_mifgsm_attack(model, images, labels, num=15,  alpha=8/255, iters=100,decay = 0.2):
    loss = nn.CrossEntropyLoss()
    # 原图像
    ori_images = images.data
    momentum = torch.zeros_like(images)
    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)
        if(outputs.argmax().item()!=labels):
            break
        model.zero_grad()
        tensor_label = torch.tensor(labels).unsqueeze(0)
        cost = loss(outputs, tensor_label)
        cost.backward()
        # 获取num点
        index_list = get_top_grad_index(images.grad.data, num, 2)
        data_grad = torch.zeros(images.grad.data.shape)
        for i in range(len(index_list)):
            data_grad[index_list[i][0],index_list[i][1],index_list[i][2]] = images.grad.data[index_list[i][0],index_list[i][1],index_list[i][2]]
        # 动量项更新
        momentum = decay * momentum + data_grad.sign()
        adv_images = images + alpha * momentum.sign()
        # 限制扰动范围
        eta = adv_images - ori_images
        # 进行下一轮对抗样本的生成。破坏之前的计算图
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
    count_df = 0
    if (outputs.argmax().item() != labels):
        point_set = torch.nonzero(ori_images-images).tolist()
        count_df = len(point_set)
        for i in point_set:
            i = tuple(i)
            if i in point_dict:

                point_dict[i] = point_dict[i] + 1
            else:
                point_dict[i] = 1
    else:
        count_df = 0

    return images,count_df



def MI_FGSM_test(raw_path):
    print("------------------MI-FGSM_test-----------------")
    model = torch.load('../train_model/model.pkl')
    AP = 0
    SR = 0
    patten = '[a-zA-Z]+'
    SR_dict = {}
    AP_dict = {}
    raw_label = []
    new_label=[]
    label_count = {}  # 统计每个预测无错的label数量
    for file in os.listdir(raw_path):  # 对一张图片开始实验
        image_path = raw_path + '/' + file
        label = re.match(patten, file).group()
        int_label = torch.tensor(label_dict[label]).argmax().item()
        raw_label.append(int_label)
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
            pertubaed_image,point = MI_FGSM_attack(model, image, int_label)
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


def RE_MI_FGSM_test(raw_path,num=15):
    print("------------------RE_MI-FGSM_test-----------------")
    model = torch.load('../train_model/model.pkl')
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
            pertubaed_image,point = re_mifgsm_attack(model, image, int_label, num)
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
        # if(count%400==0):
        #     print("raw_label")
        #     print(raw_label)
        #     print("new_label")
        #     print(new_label)
        #     print("F1-score(micro)",
        #       f1_score(raw_label, new_label, average='micro'),"F1-score(macro)",
        #       f1_score(raw_label, new_label, average='macro'))
        #     raw_label = []
        #     new_label = []
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

#输入一张图片，可视化对抗样本和原样本，同时输出不同点数结果
def resume_test(raw_path,label,num):   #raw_path: 原图片路径 label：原图片标签 num：一次攻击点的数量
    print("------------resume_test-------------")
    point_set = []
    model = torch.load('model2.pkl')
    raw_image = np.load(raw_path)
    image = copy.deepcopy(raw_image)
    image = solve_input(image)
    predict = model(image)
    pertubated_image,point = re_mifgsm_attack(model,image,label,num)
    new_predict = model(pertubated_image)
    if(new_predict.argmax()!=label and predict.argmax()==label):
        pertubated_image = pertubated_image*255
        count = 0

        for i in range(0,32):
            for j in range(0,32):
                if(pertubated_image[0][i][j]!=raw_image[i][j]):
                    point_set.append((i,j))
                    count = count+1
        pertubated_image =  torch.reshape(pertubated_image,(32,32))
        pertubated_image = pertubated_image.detach().numpy()
        print("--------pertubated_image---------")
        print(pertubated_image)
        print("------------raw_image------------")
        print(raw_image)
        print("-------different_point_set-------")
        print(point_set)
        # np.save("resume/RBitTorrent_test1", pertubated_image)
        fig = plt.figure(figsize=(16,16))  # 初始化一张画布
        plt.subplot(1,2,1)
        plt.title("pertubated image "+str(new_predict.argmax()))
        plt.imshow(pertubated_image)
        plt.subplot(1,2,2)
        plt.title("raw_image:"+str(label))
        plt.imshow(raw_image)
        plt.show()
        print("sum_different_point:",count)
    elif(predict.argmax()!=label):
        print("模型预测结果错误，无需攻击")


#输入某一类样本路径，对该类所有样本进行resume_test,key:label
def more_resume_test(raw_path,key):
    label = label_dict[key]
    label = torch.unsqueeze(torch.tensor(label).argmax(), dim=0)
    label = label.item()
    for file in os.listdir(raw_path):
        image_path = raw_path+'//'+file
        resume_test(image_path,label,15)


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
            pertubaed_image,point = re_mifgsm_attack(model, image, int_label, num)
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

    book.save("MI-FGSM_field.xls")
    return None



import time
from utils.recode import recode_data
if __name__ == "__main__":

# #统计不同攻击点数的结果记录
#     raw_path = "../dataset/npy_dataset/test"
#     col = ['k', 'SR', 'AP', 'score(micro)', 'score(macro)','time_cost']
#     for i in range(5,15):
#         start = time.time()
#         rows = RE_MI_FGSM_test(raw_path, i)
#         rows = list(rows)
#         end_time = time.time()
#         rows.append(end_time-start)
#         recode_data("MI-FGSM",rows,col)

    # #统计单纯topk方法攻击15点实验结果
    # raw_path = "../dataset/npy_dataset/test"
    # start = time.time()
    # RE_MI_FGSM_test(raw_path,15)
    # end_time = time.time()
    # print("time cost:", end_time - start)

    #统计单纯topk方法攻击15点实验结果
    raw_path = "../dataset/npy_dataset/test"
    start = time.time()
    statistic_fields(raw_path,15)
    end_time = time.time()
    print("time cost:", end_time - start)