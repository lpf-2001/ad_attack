import pandas as pd
import matplotlib.pyplot as plt
import re
#输入 元组字符串列表
#输出 （1，15）-》32+15=输出47
def transform_(str_tuple):
    result = []
    for element in str_tuple:
        res = re.findall('[0-9]+',element)
        temp_sum = 0
        for j in res:
            temp_sum = int(j)+temp_sum*32
        result.append(temp_sum)
    return result

data = pd.read_excel("../PGD/PGD_field.xls",sheet_name="facebook")
PGD = transform_(data.iloc[:,0].values)
PGD_y = data.iloc[:,1].values

data = pd.read_excel("../MI-FGSM/MI-FGSM_field.xls",sheet_name="facebook")
MI_FGSM = transform_(data.iloc[:,0].values)
MI_FGSM_y = data.iloc[:,1].values

data = pd.read_excel("../union_loss/union_loss_field.xls",sheet_name="facebook")
union_loss = transform_(data.iloc[:,0].values)
union_loss_y = data.iloc[:,1].values

data = pd.read_excel("../I-FGSM/I-FGSM_field.xls",sheet_name="facebook")
I_FGSM = transform_(data.iloc[:,0].values)
I_FGSM_y = data.iloc[:,1].values

plt.figure(figsize=(15,5))

plt.subplot(141)
plt.title('facebook')
plt.bar(PGD,PGD_y,color='g',label="PGD")
plt.legend()
plt.subplot(142)
plt.title('facebook')
plt.bar(I_FGSM,I_FGSM_y,color='c',label="I-FGSM")
plt.legend()
plt.subplot(143)
plt.title('facebook')
plt.bar(MI_FGSM,MI_FGSM_y,color='b',label="MI-FGSM")
plt.legend()
plt.subplot(144)
plt.title('facebook')
plt.bar(union_loss,union_loss_y,color='r',label="union_loss")

plt.legend()
plt.show()