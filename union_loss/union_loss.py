from __future__ import print_function

import copy
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import xlwt

import os
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from utils.utils import get_top_grad_index




point_dict = {}




def solve_input(raw_path):
    x = np.load(raw_path)
    x = x.astype("float32")
    x = x/255
    X = torch.tensor(x)
    X = torch.unsqueeze(X,0)
    X.requires_grad = True
    return X

cross_loss = nn.CrossEntropyLoss()
#label:int
def one_file_test(model,data, X,label,iters = 100,alpha=8,num=15,flag=1):
    it = 0
    new = label

    one_hot_label = torch.nn.functional.one_hot(torch.tensor(label), num_classes= 10)
    decay = 0.2
    momentum = torch.zeros_like(data)
    while(new==label and it<iters):
        output = model(X)

        real = torch.sum((one_hot_label) * output, -1)
        other = torch.maximum(torch.max((1 - one_hot_label) * output - (one_hot_label * 10000)), torch.tensor(1.0))
        # loss1 = torch.maximum(torch.tensor(-0.5), real - other + .01)   #C&W 的loss
        # real = torch.sum((one_hot_label) * output, 0)

        # other = (1 - one_hot_label) * output #- one_hot_label

        # loss = -cross_loss(output,torch.tensor([label]))
        # loss = C*loss1
        loss = real-other-cross_loss(output,torch.tensor([label]))
        # print(loss)
        loss.backward(torch.ones_like(loss),retain_graph=True)
        index_list = get_top_grad_index(X.grad.data,num,2)
        data_grad = torch.zeros(X.grad.data.shape)
        for i in range(len(index_list)):
            data_grad[index_list[i][0], index_list[i][1], index_list[i][2]] = X.grad.data[
                index_list[i][0], index_list[i][1], index_list[i][2]]

        momentum = decay*momentum + data_grad.sign()
        X = X-alpha*momentum.sign()
        X = torch.clamp(X,0,1)
        new = model(X)
        new = new.argmax().item()
        it = it+1
        X = X.detach()
        X.requires_grad = True
    if(new!=label):
        point_set = torch.nonzero((data-X)).tolist()
        point = len(point_set)
    else:
        point = 0

    if(flag==1):
        return new, point
    elif(flag==2):
        return X

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

patten = '[a-zA-Z]+'
def more_file_test(raw_path,num=15):  #求包含所有数据包的函数

    label_count = {}  #统计每个label数量
    model = torch.load("model.pkl")
    SR_dict = {}
    AP_dict = {}
    raw_label = []
    new_label=[]
    count = 0
    for file in tqdm(os.listdir(raw_path)):
        count = count+1
        label = re.match(patten, file).group()
        path = raw_path + "//" + file
        data = solve_input(path)
        predict = model(data)
        int_label = torch.tensor(label_dict[label]).argmax(axis=0).item()
        raw_label.append(int_label)
        if(predict.argmax(axis=1).item()==int_label):

            if(label in label_count):
                label_count[label] = label_count[label]+1
            else:
                label_count[label] = 1
            X = solve_input(path)
            newlabel,point = one_file_test(model,X,data,int_label,num=num)
            new_label.append(newlabel)
            if(newlabel!=int_label):
                if(label in SR_dict):
                    SR_dict[label] = SR_dict[label]+1
                else:
                    SR_dict[label] = 1
                if(label in AP_dict):
                    AP_dict[label] = AP_dict[label]+point
                else:
                    AP_dict[label] = 1

        else:
            new_label.append(predict.argmax(axis=1).item())

    SR = 0
    AP = 0
    for key in label_count:
        AP_dict[key] = AP_dict[key]/SR_dict[key]
        AP = AP + AP_dict[key]
        SR_dict[key] = SR_dict[key]/label_count[key]
        SR = SR + SR_dict[key]
        print("label",key,"count",label_count[key],"SR",SR_dict[key],"AP",AP_dict[key],"F1-score(micro)",f1_score(raw_label, new_label, average='micro'),"F1-score(macro)",f1_score(raw_label, new_label, average='macro'))
    print('ALL_SR:',SR/10,'ALL_AP:',AP/10)
    return num,SR/10,AP/10,f1_score(raw_label, new_label, average='micro'),f1_score(raw_label, new_label, average='macro')



def topktest():

    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('union_loss', cell_overwrite_ok=True)

    col = ['k','SR','AP','score(macro)','score(micro)']
    write_row = 0
    SR_list = []
    AP_list = []
    for i in range(0, 5):
        sheet.write(0, i, col[i])
    k = 20
    SR = 0
    while(k<=30):
        k = k+1
        print(k)
        a=more_file_test("USTC-TFC2016To10/test",k)
        SR_list.append(a[0])
        SR = a[0]
        AP_list.append(a[1])
        write_row = write_row+1
        sheet.write(write_row, 0 , k)
        for t in range(1,len(a)+1):
            sheet.write(write_row,t,a[t-1])
    savepath = 'union_loss2.xlsx'
    book.save(savepath)
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.plot(SR_list,'r')
    plt.subplot(1,2,2)
    plt.plot(AP_list,'b')
    plt.show()


map_packet=['BitTorrent','Facetime' ,'FTP','Gmail','MySQL','Outlook' ,'Skype','SMB' ,'Virut','Weibo']
def one_type_packet(raw_path,type,book):

    sheet = book.add_sheet(type, cell_overwrite_ok=True)

    label_count = {}  #统计每个label数量
    model = torch.load("model2.pkl")
    AP_dict = {}
    count = 0
    for file in tqdm(os.listdir(raw_path)):
        count = count+1
        label = re.match(patten, file).group()
        path = raw_path + "//" + file
        data = solve_input(path)
        predict = model(data)
        int_label = torch.tensor(label_dict[label]).argmax(axis=0).item()
        if(predict.argmax(axis=1).item()==int_label):
            if(label in label_count):
                label_count[label] = label_count[label]+1
            else:
                label_count[label] = 1
            X = solve_input(path)

            adversial = one_file_test(model,X,data,int_label,flag=2)
            newlabel = model(adversial).argmax().item()
            if(newlabel!=int_label):
                point = torch.nonzero(adversial - data).tolist()
                for i in point:
                    i = tuple(i)
                    if(i in AP_dict):
                        AP_dict[i] = AP_dict[i]+1
                    else:
                        AP_dict[i] = 1
    b = sorted(AP_dict.items(), key=lambda a: a[1], reverse=True)
    count = 0
    for i in b:
        print(i)
        sheet.write(count, 0, str(i[0]))
        sheet.write(count, 1, i[1])
        count = count+1
        if(count>15):
            break
    print("done!")


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
        # preprocess input
        image = solve_input(image_path)
        image_ = copy.deepcopy(image)
        predict = model(image)
        if(label not in label_result):
            label_result[label]={}
        if (predict.argmax() == int_label):
            pertubaed_image = one_file_test(model, image ,image_, int_label, num,flag=2)
            a = torch.nonzero(pertubaed_image-image)
            for i in range(0,len(a)):
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

    book.save("union_loss_field.xls")
    return None




import time
from utils.recode import recode_data
if __name__=="__main__":
    # Statistics_Field()
    raw_path = "../dataset/npy_dataset/test/facebook_Audio_test8.npy"
    # col = ['k', 'SR', 'AP', 'score(micro)', 'score(macro)','time_cost']
    # for i in range(5,16):
    #     start = time.time()
    #     rows = more_file_test(raw_path, i)
    #     rows = list(rows)
    #     end_time = time.time()
    #     print(type(rows))
    #     rows.append(end_time-start)
    #     recode_data("union_loss",rows,col)
    # start = time.time()
    # more_file_test(raw_path, 15)
    # end_time = time.time()
    # print("time cost:",end_time-start)

    # start = time.time()
    # statistic_fields(raw_path,15)
    # end_time = time.time()
    # print("time cost:",end_time-start)
    model = torch.load("model.pkl")
    data = solve_input(raw_path)
    data_copy = copy.deepcopy(data)
    x = one_file_test(model, data,data_copy, torch.tensor(2),flag=2)
    x = torch.reshape(x,(32,32)).detach().numpy()
    plt.imshow(x)
    plt.show()
