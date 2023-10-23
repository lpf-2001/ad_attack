import numpy as np

from tqdm import tqdm
import re
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score


def cw_l2_attack(model, images, labels, targeted=False, c=10, kappa=0, max_iter=100, learning_rate=0.01):
    images = images.to(device)
    labels = labels.to(device)

    # Define f-function
    def f(x):

        outputs = model(x)
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)

        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.byte().bool())

        # If targeted, optimize for making the other class most likely
        if targeted:
            return torch.clamp(i - j, min=-kappa)

        # If untargeted, optimize for making the other class most likely
        else:
            return torch.clamp(j - i, min=-kappa)

    w = torch.zeros_like(images, requires_grad=True).to(device)

    optimizer = optim.Adam([w], lr=learning_rate)

    prev = 1e10

    for step in range(max_iter):

        a = 1 / 2 * (nn.Tanh()(w) + 1)

        loss1 = nn.MSELoss(reduction='sum')(a, images)
        loss2 = torch.sum(c * f(a))

        cost = loss1 + loss2

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Early Stop when loss does not converge.
        if step % (max_iter // 10) == 0:
            if cost > prev:
                print('Attack Stopped due to CONVERGENCE....')
                return a
            prev = cost

        print('- Learning Progress : %2.2f %%        ' % ((step + 1) / max_iter * 100), end='\r')

    attack_images = 1 / 2 * (nn.Tanh()(w) + 1)
    return attack_images



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

if __name__ =="__main__":

    device = "cpu"

    patten = '[a-zA-Z]+'
    raw_path = '../dataset/npy_dataset/test'
    label_count = {}  # 统计每个label数量
    model = torch.load("../train_model/model.pkl")
    SR_dict = {}
    AP_dict = {}
    raw_label = []
    new_label = []
    count = 0
    start_time = time.time()
    for file in tqdm(os.listdir(raw_path)):
        # print(file)
        count = count + 1
        label = re.match(patten, file).group()
        int_label = label_dict[label]
        if (label in label_count):
            label_count[label] = label_count[label] + 1
        else:
            label_count[label] = 1
        int_label = torch.tensor(int_label).argmax().item()
        raw_label.append(int_label)
        image_path = raw_path + '//' + file
        image = np.load(image_path)
        image = torch.tensor(image).unsqueeze(0)
        image = image / 255
        predict = model(image).argmax().item()
        if (predict == int_label):
            tensor_label = torch.unsqueeze(torch.tensor(int_label), 0)
            x_adv = cw_l2_attack(model, image, torch.tensor(int_label), targeted=False, c=0.1)  # y tensor
            new_predict = model(x_adv).argmax().item()
            new_label.append(int(new_predict))
            # print(new_predict)
            if (new_predict != predict):
                point = torch.nonzero(x_adv - image)
                if (label in SR_dict):
                    SR_dict[label] = SR_dict[label] + 1
                else:
                    SR_dict[label] = 1
                if (label in AP_dict):
                    AP_dict[label] = AP_dict[label] + len(point)
                else:
                    AP_dict[label] = len(point)
        else:
            new_label.append(predict)

    SR = 0
    AP = 0
    for key in label_count:
        AP_dict[key] = AP_dict[key] / SR_dict[key]
        SR_dict[key] = SR_dict[key] / label_count[key]
        SR = SR + SR_dict[key]
        AP = AP + AP_dict[key]
        print("label", key, "count", label_count[key], "SR", SR_dict[key], "AP", AP_dict[key])
    print("F1-score(micro)", f1_score(raw_label, new_label, average='micro'), "ALL_SR", SR / 10, "ALL_AP", AP / 10)

    print("time cost:", time.time() - start_time)


