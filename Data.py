import pandas as pd
import numpy as np
"""
创建一个类，是data,有几种主要方法
把所有数据分为训练集和验证集，按照4：1的分布随机选取
data.next_batch(BATCH)是获得下一个batch 
data.train_x_miss() 是获得训练数据的属性集中测试集没有的字段
data.train_x()  是获得训练数据的属性集的正常字段
data.train_y() 是获得训练数据的预测的时间
data.test_x()  是获得验证数据的属性集
data.test_y()  是获得验证数据
在训练集数据中，有一些字段在测试集中是没有的
"""
class data():
    def __init__(self,filename):
        self.filename = filename
        with open(self.filename, "r") as f:
            self.Lines = f.readlines()
        self.index = int(len(self.Lines)/5)

    def one_hot(self,Y):
        Y_ = []
        for i in range(Y.shape[0]):
            y = [0] * 60
            y[Y[i]] = 1
            Y_.append(y)
        return np.asarray(Y_)


    def train_x_miss(self):
        x_Miss = []
        for line in self.Lines[:4*self.index]:
            line = line.rstrip(" \n").split(" ")
                #print(line)
            x_miss = line[-10:-6]+line[-4:-1]
                #print(x_miss)
            for i in range(len(x_miss)):
                x_miss[i] = int(x_miss[i])
            x_Miss.append(x_miss)
        x_Miss = np.asarray(x_Miss)
        return x_Miss

    def train_x(self):
        X = []
        for line in self.Lines[:4 * self.index]:
            line = line.rstrip(" \n").split(" ")
            # print(line)
            x = line[1:4]+line[5:-10] + line[-6:-4]
            # print(x_miss)
            for i in range(len(x)):
                x[i] = int(x[i])
            X.append(x)
        X = np.asarray(X)
        return X


    def train_y(self):
        Y = []
        for line in self.Lines[:self.index*4]:
            line = line.rstrip(" \n").split(" ")
            y = int(line[-1])
            Y.append(y)
        Y = self.one_hot(np.asarray(Y))
        #print(np.max(Y))

        return Y

    def test_x(self):
        X = []
        for line in self.Lines[4 * self.index:]:
            line = line.rstrip(" \n").split(" ")
            # print(line)
            x = line[:-10] + line[-6:-4]
            # print(x_miss)
            for i in range(len(x)):
                x[i] = int(x[i])
            X.append(x)
        X = np.asarray(X)
        return X

    def test_x_miss(self):
        x_Miss = []
        for line in self.Lines[4*self.index:]:
            line = line.rstrip(" \n").split(" ")
                #print(line)
            x_miss = line[-10:-6]+line[-4:-1]
                #print(x_miss)
            for i in range(len(x_miss)):
                x_miss[i] = int(x_miss[i])
            x_Miss.append(x_miss)
        x_Miss = np.asarray(x_Miss)
        return x_Miss

    def test_y(self):
        Y = []
        for line in self.Lines[self.index * 4:]:
            line = line.rstrip(" \n").split(" ")
            y = int(line[-1])
            Y.append(y)
        Y = self.one_hot(np.asarray(Y))

        return Y

    def next_batch(self,BATCH_SIZE,step):
        x = self.train_x()
        y = self.train_y()
        s = int(len(y) / BATCH_SIZE)
        index = (step % s)*BATCH_SIZE

        return x[index:index+BATCH_SIZE],y[index:index+BATCH_SIZE]

