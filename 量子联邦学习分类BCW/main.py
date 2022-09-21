import time

import pennylane as qml
import torch
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import datetime
import sys

Begin_time = time.time()
np.random.seed(0)
torch.manual_seed(0)

num_classes = 2
margin = 0.15
feature_size = 9
batch_size = 10
# lr_adam = 0.01
lr=0.01
momentum = 0.9
train_split = 1
# the number of the required qubits is calculated from the number of features
num_qubits = int(np.ceil(np.log2(feature_size)))
num_layers = 6
total_iterations = 100
dev = qml.device("default.qubit", wires=num_qubits)

def layer(W):
    for i in range(num_qubits):
        #每个qubit做一个任意的旋转，参数是3个，那么旋转门的参数个数=3*num_qubit
        qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
        #电路层由每个qubit的任意旋转以及将每个qubit与其相邻的qubit的CNOT门组成
        #这里是可以调整的，建议继续用CNOT但是可以改一下关联
    for j in range(num_qubits - 1):
        qml.CNOT(wires=[j, j + 1])
        #qml.CZ(wires=[j,j+1])
        # qml.CY(wires=[j,j+1])
    if num_qubits >= 2:
        # Apply additional CNOT to entangle the last with the first qubit
        qml.CNOT(wires=[num_qubits - 1, 0])
        #qml.CZ(wires=[num_qubits-1,0])
        # qml.CY(wires=[num_qubits-1,0])
def circuit(weights, feat=None):
    #normalize：归一化
    qml.AmplitudeEmbedding(feat, range(num_qubits), pad_with=0.0, normalize=True)
    #采用振幅编码对feature进行编码，由于正好是四个feature就不需要pad（如果维度不够的话，将维度扩张至符合要求）
    for W in weights:
        layer(W)#传入的weight对电路进行初始化
    return qml.expval(qml.PauliZ(0)) #最后对第一个qubit做Pauliz测量得到分类的结果

qnodes = []
#一个qnode负责把一个类分出去，这里有三个类所以需要三个qnode
for iq in range(num_classes):
    qnode = qml.QNode(circuit, dev, interface="torch")
    qnodes.append(qnode)

def variational_classifier(q_circuit, params, feat):
#构造变分分类器
    weights = params[0]
    bias = params[1]
    return q_circuit(weights, feat=feat) + bias

def multiclass_svm_loss(q_circuits, all_params, feature_vecs, true_labels):
#多分类的损失函数
    loss = 0
    num_samples = len(true_labels)
    for i, feature_vec in enumerate(feature_vecs):
        #将样本编号
        # Compute the score given to this sample by the classifier corresponding to the
        # true label. So for a true label of 1, get the score computed by classifer 1,
        # which distinguishes between "class 1" or "not class 1".
        s_true = variational_classifier(
            q_circuits[int(true_labels[i])],
            (all_params[0][int(true_labels[i])], all_params[1][int(true_labels[i])]),
            feature_vec,)
        s_true = s_true.float()
        li = 0

        # Get the scores computed for this sample by the other classifiers
        for j in range(num_classes):
            if j != int(true_labels[i]):
                s_j = variational_classifier(
                    q_circuits[j], (all_params[0][j], all_params[1][j]), feature_vec
                )
                s_j = s_j.float()
                li += torch.max(torch.zeros(1).float(), s_j - s_true + margin)
        loss += li

    return loss / num_samples

def classify(q_circuits, all_params, feature_vecs, labels):
    predicted_labels = []
    for i, feature_vec in enumerate(feature_vecs):
        scores = np.zeros(num_classes)
        for c in range(num_classes):
            score = variational_classifier(
                q_circuits[c], (all_params[0][c], all_params[1][c]), feature_vec
            )
            scores[c] = float(score)
        pred_class = np.argmax(scores)
        #返回值最大的索引，得分最高的那一类就是结果
        predicted_labels.append(pred_class)
    return predicted_labels

def accuracy(labels, hard_predictions):
    loss = 0
    for l, p in zip(labels, hard_predictions):
        if torch.abs(l - p) < 1e-5:
            loss = loss + 1 #预测正确的样本数加一
    loss = loss / labels.shape[0]
    return loss

def load_and_process_data():
    #加载数据
    data = np.loadtxt("./dataset/Breast Cancer Wisconsin_shuffle.csv", delimiter=",")
    X = torch.tensor(data[:, 0:feature_size])#feature
    #print("First X sample, original  :", X[0])

    # normalize each input
    normalization = torch.sqrt(torch.sum(X ** 2, dim=1))
    #将数据标准化，防止单位的不同导致的问题
    X_norm = X / normalization.reshape(len(X), 1)
    #print("First X sample, normalized:", X_norm[0])

    Y = torch.tensor(data[:, -1])#标签
    return X, Y

# Create a train and test split.
#把数据集划分为训练集和测试集
def split_data(feature_vecs, Y):
    num_data = len(Y)
    num_train = int(train_split * num_data)
    index = np.random.permutation(range(num_data))
    feat_vecs_train = feature_vecs[index[:num_train]]
    Y_train = Y[index[:num_train]]
    feat_vecs_test = feature_vecs[index[num_train:]]
    Y_test = Y[index[num_train:]]
    return feat_vecs_train, feat_vecs_test, Y_train, Y_test

def training(features, Y):
    num_data = Y.shape[0]
    feat_vecs_train, feat_vecs_test, Y_train, Y_test = split_data(features, Y)
    num_train = Y_train.shape[0]
    q_circuits = qnodes

    # Initialize the parameters
    all_weights = [
        Variable(0.1 * torch.randn(num_layers, num_qubits, 3), requires_grad=True)
        for i in range(num_classes)
    ]
    all_bias = [Variable(0.1 * torch.ones(1), requires_grad=True) for i in range(num_classes)]
    params = (all_weights, all_bias)
    optimizer = optim.Adam(all_weights + all_bias, lr=lr)
    # optimizer = optim.SGD(all_weights+all_bias,lr,momentum)
    # optimizer  = optim.ASGD(all_weights+all_bias,lr)
    # optimizer = optim.RAdam(all_weights+all_bias,lr=lr)
    costs, train_acc, test_acc = [], [], []

    today = datetime.datetime.today()
    todaystr = today.strftime('%Y-%m-%d')
    print("-----------------------------Begin Training-----------------------------")
    # train the variational classifier
    for it in range(total_iterations):
        batch_index = np.random.randint(0, num_train, (batch_size,))
        feat_vecs_train_batch = feat_vecs_train[batch_index]
        Y_train_batch = Y_train[batch_index]
        optimizer.zero_grad()
        curr_cost = multiclass_svm_loss(q_circuits, params, feat_vecs_train_batch, Y_train_batch)
        curr_cost.backward()
        optimizer.step()

        # Compute predictions on train and validation set
        predictions_train = classify(q_circuits, params, feat_vecs_train, Y_train)
        # predictions_test = classify(q_circuits, params, feat_vecs_test, Y_test)
        acc_train = accuracy(Y_train, predictions_train)
        # acc_test = accuracy(Y_test, predictions_test)

        # print(
        #     "Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Acc test: {:0.7f} "
        #     "".format(it + 1, curr_cost.item(), acc_train, acc_test)
        # )
        log_path= "./log/"+todaystr+" log.txt"
        print('--------------------------Write InFo--------------------------------')
        with open(log_path, "a+") as file:
            # file.write(
            #     "Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f}\n "
            #     "".format(it + 1, curr_cost.item(), acc_train, acc_test)
            # )
            file.write(
                "Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f}\n "
                "".format(it + 1, curr_cost.item(), acc_train)
            )
        #print(params)
        if it==total_iterations-1:
            params_path ="./param/"+todaystr+" param.txt"
            print("--------------------------Write Params------------------------------")
            __console = sys.stdout
            fp=open(params_path,"a+")
            sys.stdout=fp
            print("--------------------------all_weights-----------------------------")
            print(all_weights)
            print("---------------------------all_bias-------------------------------")
            print(all_bias)
            print("---------------------------all_param-------------------------------")
            print(params)
            sys.stdout = __console
        costs.append(curr_cost.item())
        train_acc.append(acc_train)
        # test_acc.append(acc_test)

    # return costs, train_acc, test_acc
    return costs, train_acc

# We now run our training algorithm and plot the results. Note that
# for plotting, the matplotlib library is required

features, Y = load_and_process_data()
# costs, train_acc, test_acc = training(features, Y)
costs, train_acc = training(features,Y)

import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
iters = np.arange(0, total_iterations, 1)
colors = ["tab:red", "tab:blue"]
ax1.set_xlabel("Iteration", fontsize=17)
ax1.set_ylabel("Cost", fontsize=17, color=colors[0])
ax1.plot(iters, costs, color=colors[0], linewidth=4)
ax1.tick_params(axis="y", labelsize=14, labelcolor=colors[0])

# ax2 = ax1.twinx()
# ax2.set_ylabel("Test Acc.", fontsize=17, color=colors[1])
# ax2.plot(iters, test_acc, color=colors[1], linewidth=4)
#
# ax2.tick_params(axis="x", labelsize=14)
# ax2.tick_params(axis="y", labelsize=14, labelcolor=colors[1])

plt.grid(False)
plt.tight_layout()
plt.show()

today = datetime.datetime.today()
todaystr = today.strftime('%Y-%m-%d')
log_path = "./log/"+todaystr + " log.txt"
During_time = time.time()-Begin_time
with open(log_path,"a+") as file:
    file.write("Total Run Time:"+str(During_time)+" s\n")