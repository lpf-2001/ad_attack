import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel("../PGD/PGD.xls")
PGD_acc = data.iloc[0:,1].values
PGD_ap= data.iloc[0:,2].values
PGD_time = data.iloc[0:,5].values
data = pd.read_excel("../MI-FGSM/MI-FGSM.xls")
MI_FGSM_acc = data.iloc[0:,1].values
MI_FGSM_ap = data.iloc[0:,2].values
MI_FGSM_time = data.iloc[0:,5].values
data = pd.read_excel("../I-FGSM/I_FGSM.xls")
I_FGSM_acc = data.iloc[0:,1].values
I_FGSM_ap = data.iloc[0:,2].values
I_FGSM_time = data.iloc[0:,5].values
data = pd.read_excel("../union_loss/union_loss.xls")
union_loss_acc = data.iloc[0:,1].values
union_loss_ap = data.iloc[0:,2].values
union_loss_time = data.iloc[0:,5].values
data = pd.read_excel("../MJSMA/JSMA.xls")
jsma_acc = data.iloc[0:,1].values
jsma_ap = data.iloc[0:,2].values
jsma_time = data.iloc[0:,5].values

a = []
for i in range(5,16):
    a.append(i)

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.title("attack accuracy")
plt.plot(a,union_loss_acc,'r',label="union_attack")
plt.plot(a,MI_FGSM_acc,'b',label="MI-FGSM_attack")
plt.plot(a,PGD_acc,'g',label="PGD_attack")
plt.plot(a,jsma_acc,'y',label="JSMA_attack")
plt.plot(a,I_FGSM_acc,'c',label="I-FGSM_attack")
plt.legend()

plt.subplot(1,3,2)
plt.title("attack average point")
plt.plot(a,union_loss_ap,'r',label="union_loss_average_point")
plt.plot(a,PGD_ap,'g',label="PGD_average_point")
plt.plot(a,MI_FGSM_ap,'b',label="MI-FGSM_average_point")
plt.plot(a,jsma_ap,'y',label="JSMA_average_point")
plt.plot(a,I_FGSM_ap,'c',label="I-FGSM_average_point")
plt.legend()

plt.subplot(1,3,3)
plt.title("attack time cost")
plt.plot(a,union_loss_time,'r',label="union_loss_time")
plt.plot(a,PGD_time,'g',label="PGD_time")
plt.plot(a,MI_FGSM_time,'b',label="MI-FGSM_time")
plt.plot(a,jsma_time,'y',label="JSMA_time")
plt.plot(a,I_FGSM_time,'c',label="I-FGSM_time")
plt.legend()
plt.show()