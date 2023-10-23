import numpy as np
import torch
import torchattacks
import torch
from tqdm import tqdm
import numpy as np
import os
import re
from sklearn.metrics import f1_score
import time
torch.autograd.set_detect_anomaly(True)
MAX_ITERATIONS = 10  # number of iterations to perform gradient descent
ABORT_EARLY = True  # abort gradient descent upon first valid solution
LEARNING_RATE = 1e-2  # larger values converge faster to less accurate results
INITIAL_CONST = 1e-3  # the first value of c to start at
LARGEST_CONST = 20  # the largest value of c to go up to before giving up
REDUCE_CONST = False  # try to lower c each iteration; faster to set to false
TARGETED = False  # should we target one specific class? or just be wrong?
CONST_FACTOR = 2.0  # f>1, rate at which we increase constant, smaller better


class CarliniL0():
    def __init__(self, model, num_labels=10, image_size=32, num_channels=1, batch_size=1,
                 targeted=TARGETED, learning_rate=LEARNING_RATE,
                 max_iterations=MAX_ITERATIONS, abort_early=ABORT_EARLY,
                 initial_const=INITIAL_CONST, largest_const=LARGEST_CONST,
                 reduce_const=REDUCE_CONST, const_factor=CONST_FACTOR,
                 independent_channels=False):
        """
        The L_0 optimized attack.
        Returns adversarial examples for the supplied model.
        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. Should be set to a very small
          value (but positive).
        largest_const: The largest constant to use until we report failure. Should
          be set to a very large value.
        const_factor: The rate at which we should increase the constant, when the
          previous constant failed. Should be greater than one, smaller is better.
        independent_channels: set to false optimizes for number of pixels changed,
          set to true (not recommended) returns number of channels changed.
        """

        self.model = model
        self.num_labels = num_labels
        self.image_size = image_size
        self.num_channels = num_channels
        self.batch_size = batch_size

        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.ABORT_EARLY = abort_early
        self.INITIAL_CONST = initial_const  # c
        self.LARGEST_CONST = largest_const
        self.REDUCE_CONST = reduce_const
        self.const_factor = const_factor
        self.independent_channels = independent_channels

        self.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK = False

    def doit(self, oimgs, labs, starts, valid, CONST):
        # convert to tanh-space

        imgs = torch.atanh(oimgs)
        starts = torch.atanh(starts)

        shape = (self.batch_size, self.num_channels, self.image_size, self.image_size)

        # the variable to optimize over
        modifier = torch.nn.parameter.Parameter(torch.zeros(shape, dtype=torch.float32, device=device),
                                                requires_grad=True)

        # the variables we're going to hold, use for efficiency
        canchange = valid.to(device)
        simg = starts.to(device)
        original = oimgs.to(device)
        timg = imgs.to(device)
        tlab = labs.to(device)
        const = []

        tlab = torch.nn.functional.one_hot(tlab, num_classes=10)

        optimizer = torch.optim.Adam([modifier], lr=self.LEARNING_RATE)

        while CONST < self.LARGEST_CONST:
            # try solving for each value of the constant
            # print('try const', CONST)
            for step in range(self.MAX_ITERATIONS):

                # remember the old value
                oldmodifier = modifier

                newimg = (torch.tanh(modifier + simg) / 2) * canchange + (1 - canchange) * original

                output = self.model(newimg)

                real = torch.sum((tlab) * output, -1)
                other = torch.maximum(torch.max((1 - tlab) * output - (tlab * 10000)), torch.tensor(1.0, device=device))

                if self.TARGETED:
                    # if targetted, optimize for making the other class most likely
                    loss1 = torch.maximum(torch.tensor(0.0, device=device), other - real + .01)
                else:
                    # if untargeted, optimize for making this class least likely.
                    loss1 = torch.maximum(torch.tensor(0.0, device=device), real - other + .01)

                # sum up the losses
                loss2 = torch.sum(torch.square(newimg - torch.tanh(timg) / 2))
                loss = CONST * loss1 + loss2

                works = loss1.item()
                scores = output#.detach()
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                if torch.all(scores >= -.0001) and torch.all(scores <= 1.0001):
                    if torch.allclose(torch.sum(scores, axis=1), 1.0, atol=1e-3):
                        if not self.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK:
                            raise Exception(
                                "The output of model.predict should return the pre-softmax layer. It looks like you are returning the probability vector (post-softmax). If you are sure you want to do that, set attack.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK = True")

                if works < .0001 and self.ABORT_EARLY:
                    modifier = oldmodifier
                    grads = torch.autograd.grad(loss, modifier, torch.ones_like(loss))[0]
                    # grads = grads.detach()
                    nimg = (torch.tanh(modifier + simg) / 2) * canchange + (1 - canchange) * original
                    nimg = nimg.detach()
                    return grads, scores, nimg, CONST

            # we didn't succeed, increase constant and try again
            CONST *= self.const_factor

    def attack(self, imgs, targets):
        """
        Perform the L_0 attack on the given images for the given targets.
        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = torch.tensor([], device=device)
        for i, (img, target) in enumerate(zip(imgs, targets)):
            # print("Attack iteration",i)
            r = torch.cat((r, self.attack_single(img, target)), dim=0)
        return r

    def attack_single(self, img, target):
        """
        Run the attack on a single image and label
        """

        # the pixels we can change
        valid = torch.ones((self.batch_size, self.num_channels, self.image_size, self.image_size))

        # the previous image
        prev = img.clone().reshape((self.batch_size, self.num_channels, self.image_size, self.image_size))

        # initially set the solution to None, if we can't find an adversarial
        # example then we will return None as the solution.
        last_solution = img.unsqueeze(0)
        const = self.INITIAL_CONST

        equal_count = None

        while True:
            res = self.doit(img.clone(), target, prev.clone(),
                            valid, const)

            if(model(res[2]).argmax().item()!=int(target)):
                return res[2]

            if res == None:
                # the attack failed, we return this as our final answer
                # print("Final answer",equal_count)
                return last_solution

            # the attack succeeded, now we pick new pixels to set to 0
            restarted = False
            gradientnorm, scores, nimg, const = res
            if self.REDUCE_CONST: const /= 2

            equal_count = self.num_channels * self.image_size ** 2 - torch.sum(
                torch.all(torch.abs(img - nimg[0]) < .0001, dim=0, keepdim=True))
            # print("Forced equal:",torch.sum(1-valid),
            #       "Equal count:",equal_count)
            if torch.sum(valid) == 0:
                # if no pixels changed, return
                return img.unsqueeze(0)

            if self.independent_channels:
                # we are allowed to change each channel independently
                valid = valid.flatten()
                totalchange = torch.abs(nimg[0] - img) * torch.abs(gradientnorm[0])
            else:
                # we care only about which pixels change, not channels independently
                # compute total change as sum of change for each channel
                valid = valid.flatten()
                totalchange = torch.abs(torch.sum(nimg[0] - img, axis=0)) * torch.sum(torch.abs(gradientnorm[0]),axis=0)
            totalchange = totalchange.flatten()

            # set some of the pixels to 0 depending on their total change
            did = 0
            for e in torch.argsort(totalchange):
                if torch.all(valid[e]):
                    did += 1
                    valid[e] = 0

                    if totalchange[e] > .01:
                        # if this pixel changed a lot, skip
                        break
                    if did >= .3 * equal_count ** .5:
                        # if we changed too many pixels, skip
                        break

            valid = torch.reshape(valid, (self.batch_size, self.num_channels, self.image_size, self.image_size))
            # print("Now forced equal:",torch.sum(1-valid))

            last_solution = prev = nimg


device = "cpu"
model = torch.load('../train_model/model.pkl')
atk = CarliniL0(model, num_labels = 10, image_size=32, num_channels=1, batch_size=1, max_iterations=100, initial_const=1, largest_const=10)
# image = np.load("USTC-TFC2016To10/new_test/BitTorrent_train176.jpg.npy")
# image = torch.tensor(image).unsqueeze(0)
#
# image = image/255
# x_adv = atk.attack_single(image,torch.tensor(0))
# print(x_adv)
# print((x_adv==image).all())
# print(model(x_adv).argmax())





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
    count = count + 1
    label = re.match(patten, file).group()
    int_label = label_dict[label]

    int_label = torch.tensor(int_label).argmax().item()
    raw_label.append(int_label)
    image_path = raw_path + '//' + file
    image = np.load(image_path)
    image = torch.tensor(image).unsqueeze(0)
    image = image / 255
    predict = model(image).argmax().item()
    if(predict == int_label):
        if (label in label_count):
            label_count[label] = label_count[label] + 1
        else:
            label_count[label] = 1
        tensor_label = torch.unsqueeze(torch.tensor(int_label),0)
        x_adv = atk.attack_single(image, torch.tensor(int_label))          #y tensor
        new_predict = model(x_adv).argmax().item()
        new_label.append(int(new_predict))
        # print(new_predict)
        if(new_predict!=predict):
            point = torch.nonzero(x_adv-image)
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

print("time cost:",time.time()-start_time)
print(raw_label)
print(new_label)