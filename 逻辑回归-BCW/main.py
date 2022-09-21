import numpy as np
def perdict(w,x): #根据权重和逻辑函数求解最新的预测值，x为单个数据
    if len(w) != len(x):
        print("w和x的维度不一致")
        return None
    z=0
    for index in range(len(w)):
       z = z + w[index]*x[index]
    y = (1+np.e**-(z))**-1
    return y

def upadteWeight(oldW, data, predict, label): #按照梯度下降的原则更新权重
    offset = np.array([np.array(data[i]) * (label[i] - j) for i,j in enumerate(predict)])
    offset = np.array([np.sum(offset[...,index]) for index in range(len(w))])/len(data)
    newW = oldW + offset
    return newW
if __name__ == '__main__':
    data = [[1,0,1],[2,0,1],[3,0,1],[8,9,1],[9,9,1],[10,9,1]]
    label = [0, 0, 0, 1, 1, 1]
    w = [0,0,0]
    while True:
        predicts = [perdict(w,index) for index in data]
        w=upadteWeight(w, data, predicts, label)
        if sum(predicts[:3])<0.5 and sum(predicts[3:])>2: #设置更新退出条件，前三个的预测值足够第（代表类别为0），后三个预测足够高（类别为1）
            break;
    print("w:", w)
    print("predict:", predicts)
    print(perdict(w,[10,8,1]))
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer # 乳腺癌数据集
from sklearn.model_selection import train_test_split
if __name__ == '__main__':
    data=load_breast_cancer() #加载数据集
    X =data['data'] #数据
    Y = data['target'] #标签
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=1) #划分测试集和训练集
    l1 = LogisticRegression(penalty="l1",C=0.5,solver="liblinear") #配置逻辑回归，penalty为正则化，solver为求解w的方法
    l1.fit(X_train,Y_train)
    score =  l1.score(X_test,Y_test)
    print(score)
