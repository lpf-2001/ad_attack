import torch
import copy
import numpy as np




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