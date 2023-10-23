import numpy as np
import torch
import re
from torch.autograd import Variable
import xlwt
import matplotlib.pyplot as plt
# from torch.autograd.gradcheck import zero_gradients
import os
import copy
import time

from sklearn.metrics import f1_score
from tqdm import tqdm

def solve_input(image): #处理输入归一化,输入是(32,32)的未归一化的numpy图片，输出是(1,32,32)的tensor
    image = image.astype(np.float32)
    image = image / 255
    image = torch.tensor(image)
    image = image.reshape((1,32,32))
    image.requires_grad = True
    return image


def compute_jacobian(model, input):
    output = model(input)
    num_features = int(np.prod(input.shape[1:]))
    jacobian = torch.zeros([output.size()[1], num_features])
    mask = torch.zeros(output.size())
    # mask = mask.to('cuda')
    for i in range(output.size()[1]):
        mask[:, i] = 1
        # input.grad.zero_()
        output.backward(mask, retain_graph=True)
        # print(input.grad)
        jacobian[i] = input.grad.reshape((-1, num_features)).clone()
        mask[:, i] = 0
    return jacobian

def max_perturbation_choose(Vector, gamma):
    mid_number = int(Vector.shape[1] * gamma)
    abs_vector = torch.abs(Vector)
    for i in range(0,1024):
        if(i>=34):
            break
        abs_vector[0,i]=-1
    sort_vector = abs_vector.sort(1, True)[0]
    mid_value = (sort_vector[:, mid_number - 2] + sort_vector[:, mid_number - 1]) / 2
    one_matrix = torch.ones_like(Vector)
    shape_num = Vector.shape[0]
    compare_matrix = one_matrix * mid_value.view(shape_num, 1)
    Mask_range = abs_vector.gt(compare_matrix)
    # print("Mask_range",len(torch.nonzero(Mask_range)),"gamma",gamma,"Vectorshape",Vector.shape[1],"mid_number",gamma*Vector.shape[1])
    Vector_range = Vector * Mask_range
    return Vector_range

def saliency_map(jacobian, target_index, nb_features, targeted):
    all_sum = torch.sum(jacobian, dim=0, keepdim=True)
    target_grad = jacobian[target_index]
    others_grad = all_sum - target_grad
    adv_temp_map = others_grad * target_grad
    if targeted == "targeted":
        mask1 = target_grad.gt(0)  # find + perturbation
        mask2 = -1 * target_grad.lt(0)  # find - perturbation
    else:
        mask1 = -1 * target_grad.gt(0)  # find - perturbation
        # print("target_grad.gt",target_grad.gt(0))
        mask2 = target_grad.lt(0)  # find + perturbation
        # print("target_grad.lt",target_grad.lt(0))
    mask3 = mask1 + mask2  # merge perturbation
    # print()
    mask4 = adv_temp_map.lt(0)  # localtion perturbation
    adv_saliency_map = torch.abs(adv_temp_map) * mask4 * mask3
    return adv_saliency_map


def generate_adversarial_example(image, ys_target, gamma, model, targeted, eplision=5):
    var_sample = solve_input(image)
    var_target = Variable(torch.LongTensor([ys_target, ]))
    # print(var_target)
    num_features = int(np.prod(var_sample.shape[1:]))
    shape = var_sample.size()
    model.eval()
    output = model(var_sample)
    current = torch.max(output.data, 1)[1].numpy()
    jacobian = compute_jacobian(model, var_sample)
    adv_saliency_map = saliency_map(jacobian, var_target, num_features, targeted)
    adv_saliency_map = max_perturbation_choose(adv_saliency_map, gamma)
    var_sample = var_sample.reshape((1,1024))
    new_examples = torch.clamp(adv_saliency_map * eplision + var_sample, 0.0, 1.0)
    adversarial_examples = new_examples.view(shape)
    return adversarial_examples



 # Perturbation rate
def MJSMA(net,raw_path,label,gamma = 0.01,eplision=5):  #label 是int数值

    testdata = np.load(raw_path)
    testdata = testdata.reshape((1,1024))

    ys_target = label # Adversarial label, if untarget, ys_target is the source label.
    targeted = 'untargeted'
    copy_image = copy.deepcopy(testdata)
    copy_image = solve_input(copy_image)
    outputs = net(copy_image)
    predicted = torch.max(outputs.data, 1)[1]
    raw_predict = predicted[0]
    if(predicted[0]!=label):
        return predicted[0],0,[]
    # print('测试样本扰动前的预测值：{}'.format(predicted[0]))
    # Craft adversarial adversarial examples
    adversarial_examples = generate_adversarial_example(testdata, ys_target, gamma, net, targeted,eplision)
    outputs = net(adversarial_examples)
    predicted = torch.max(outputs.data, 1)[1]
    new_predict = predicted[0]
    i = 0
    while(i<100):
        if (new_predict != raw_predict):
            break
        adversarial_examples = adversarial_examples.reshape((1, 1024))
        adversarial_examples = adversarial_examples.detach().numpy()
        adversarial_examples = adversarial_examples*255
        adversarial_examples = generate_adversarial_example(adversarial_examples, ys_target, gamma, net, targeted, eplision)
        outputs = net(adversarial_examples)
        predicted = torch.max(outputs.data, 1)[1]
        new_predict = predicted[0]

        # print(i)
        i = i+1
    # print('测试样本扰动后的预测值：{}'.format(predicted[0]))
    count = 0
    point_set = []
    if(raw_predict!=new_predict):
        point = torch.nonzero(copy_image-adversarial_examples)
        count = len(point)
        for i in range(0,32):
            for j in range(0,32):
                if(copy_image[0,i,j]!=adversarial_examples[0,i,j]):
                    count = count+1
                    point_set.append((i,j))

    return new_predict.item(),count,point_set



def restrain(point_set):       #攻击的点集列表，成员是（a,b）
    count = 0
    for data in point_set:
        if(data[0]*32+data[1]<54):
            count = count +1
    return count
def MJSMA_test(raw_path,num,eplision=8):         #包含所有类型数据包的文件路径
    gamma = num/1024
    label_count = {}
    SR_dict = {}
    AP_dict = {}
    raw_label = []
    new_label = []
    model = torch.load('../train_model/model.pkl')
    patten = '[a-zA-Z]+'
    count = 0
    for file in tqdm(os.listdir(raw_path)):
        count = count+1
        label = re.match(patten, file).group()
        int_label = label_dict[label]
        if(label in label_count):
            label_count[label] = label_count[label]+1
        else:
            label_count[label] = 1
        int_label = torch.tensor(int_label).argmax().item()
        raw_label.append(int_label)
        image_path = raw_path+'//'+file
        new_predict,point,point_set = MJSMA(model,image_path,int_label,gamma,eplision)
        new_label.append(int(new_predict))

        if (new_predict != int_label):
            if(label in SR_dict):
                SR_dict[label] = SR_dict[label]+1
            else:
                SR_dict[label] = 1
            if(label in AP_dict):
                AP_dict[label] = AP_dict[label]+point
            else:
                AP_dict[label] = point
    SR = 0
    AP = 0
    for key in label_count:
        AP_dict[key] = AP_dict[key] / SR_dict[key]
        SR_dict[key] = SR_dict[key] / label_count[key]
        SR = SR + SR_dict[key]
        AP = AP + AP_dict[key]
        print("label", key, "count", label_count[key], "SR", SR_dict[key], "AP", AP_dict[key])
    print("F1-score(micro)",f1_score(raw_label, new_label, average='micro'),"ALL_SR",SR/10,"ALL_AP",AP/10)
    return num,SR/10,AP/10,f1_score(raw_label, new_label, average='macro'),f1_score(raw_label, new_label, average='micro')


#一类数据包的MJSMA 测试gamma和eplision
def plot_MJSMA(raw_path,label,gamma=0.03,eplision=5):   #输入：raw_path:一类数据包路径 label:该类别标签int   输出：攻击成功率，平均攻击点数，最大攻击点数
    count = 0
    max_point = 0
    sum_point = 0
    re_sum_point = 0
    success_count = 0
    model = torch.load("../model2.pkl")
    for file in os.listdir(raw_path):
        count = count+1
        image_path = raw_path+"//"+file
        new_predict,point,point_set = MJSMA(model,image_path,label,gamma,eplision)
        if(label!=new_predict and point==0):
            count = count-1
            continue
        if (new_predict!=label):
            re_point = restrain(point_set)
            re_sum_point = re_sum_point+re_point
            success_count = success_count + 1
            if (max_point < point):
                max_point = point
            sum_point = sum_point + point
    print("攻击局域外点数:",re_sum_point)
    return success_count/count,sum_point/success_count,max_point

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

def plot_gamma_RE(raw_path,label):
    success_rate_list = []
    average_point_list = []
    max_point_list =[]

    for i in range(1,11):
        gamma = i/100
        success_rate,average_point,max_point = plot_MJSMA(raw_path,label,gamma)
        print("success_rate",success_rate,"average_point",average_point,"max_point",max_point)
        success_rate_list.append(success_rate)
        average_point_list.append(average_point)
        max_point_list.append(max_point)
    i = np.arange(1,11)
    i = i/100
    plt.figure(figsize=(16,16))
    plt.subplot(1,3,1)
    plt.title("success_rate&gamma")
    plt.plot(i,success_rate_list)

    plt.subplot(1,3,2)
    plt.title("average_point&gamma")
    plt.plot(i,average_point_list)

    plt.subplot(1,3,3)
    plt.title("max_point&gamma")
    plt.plot(i,max_point_list)
    plt.savefig('MJSMA.png')
    plt.show()


def plot_ep_RE(raw_path,label):
    success_rate_list = []
    average_point_list = []
    max_point_list = []

    for i in range(1, 11):
        gamma = 0.03
        success_rate, average_point, max_point = plot_MJSMA(raw_path, label, gamma,eplision=i)
        print("success_rate", success_rate, "average_point", average_point, "max_point", max_point)
        success_rate_list.append(success_rate)
        average_point_list.append(average_point)
        max_point_list.append(max_point)
    i = np.arange(1, 11)
    plt.figure(figsize=(16, 16))
    plt.subplot(1, 3, 1)
    plt.title("success_rate&gamma")
    plt.plot(i, success_rate_list)

    plt.subplot(1, 3, 2)
    plt.title("average_point&gamma")
    plt.plot(i, average_point_list)

    plt.subplot(1, 3, 3)
    plt.title("max_point&gamma")
    plt.plot(i, max_point_list)
    plt.savefig('EP_MJSMA.png')
    plt.show()


def small_test(raw_path):
    time_start= time.time()
    patten = '[a-zA-Z]+'
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('MJSMA_small', cell_overwrite_ok=True)
    col = ['gamma', 'eplision', 'label', '平均修改点数', '最大修改点数', '样本总数', '攻击成功样本数', '攻击成功率', 'payload外修改点数']
    write_row = 1
    model = torch.load('../model2.pkl')
    for i in range(0, 9):
        sheet.write(0, i, col[i])
    for gamma in range(1,11):
        gamma = gamma/100
        for eplision in range(1,11):
            re_sum_point = 0
            success_count = 0
            count = 0
            max_point = 0
            sum_point = 0
            for file in tqdm(os.listdir(raw_path)):
                count = count+1
                label = re.match(patten, file).group()
                label = torch.tensor(label_dict[label]).argmax().item()
                print(file,label)
                image_path = raw_path+'//'+file

                new_predict, point, point_set = MJSMA(model,image_path, label, gamma, eplision)
                if (label!=new_predict and point==0):
                    count = count - 1
                    continue
                if (new_predict!=label):
                    re_point = restrain(point_set)
                    re_sum_point = re_sum_point + re_point
                    success_count = success_count + 1
                    if (max_point < point):
                        max_point = point
                    sum_point = sum_point + point
            sheet.write(write_row, 0, gamma)
            sheet.write(write_row, 1, eplision)
            sheet.write(write_row, 2, re.match(patten, file).group())
            sheet.write(write_row, 3, sum_point/success_count)
            sheet.write(write_row, 4, max_point)
            sheet.write(write_row, 5, count)
            sheet.write(write_row, 6, success_count)
            sheet.write(write_row, 7, success_count/count)
            sheet.write(write_row, 8, re_sum_point)
            write_row = write_row+1
            print( "re_Point:", re_sum_point, "average_point:", sum_point / count, "max_point:", max_point,
                  "count:", count, "success_count:", success_count, "success_rate:", success_count / count)

    savepath = 'SD_MJSMA.xls'
    book.save(savepath)
    time_end = time.time()
    print('time cost', time_end - time_start, 's')

import time
from utils.recode import recode_data

if __name__=="__main__":
    time_start = time.time()
    # MJSMA('../USTC-TFC2016To10/test/FTP_test1.jpg.npy',2)
    # raw_path = '../USTC-TFC2016To10/test/'

    # raw_path = '../dataset/npy_dataset/test'
    # MJSMA_test(raw_path,15/1024,8)         #所有类包的成功率等统计
    # print('time cost', time.time() - time_start, 's')

    raw_path = "../dataset/npy_dataset/test"
    col = ['k', 'SR', 'AP', 'score(micro)', 'score(macro)','time_cost']
    for i in range(5,16):
        start = time.time()
        rows = MJSMA_test(raw_path,i,8)
        rows = list(rows)
        end_time = time.time()
        rows.append(end_time-start)
        recode_data("JSMA",rows,col)
