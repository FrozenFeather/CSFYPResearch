import numpy as np
import scipy.io as io
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from datetime import datetime
from batch import *
import sys
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, onehot_size, embed_size):
        super(Net, self).__init__()
        self.fcA1 = nn.Linear(onehot_size, embed_size, bias=False)

        self.fcB1 = nn.Linear(embed_size, onehot_size, bias=False)

    def forward(self, x):
        embed = self.fcA1(x)

        pred = self.fcB1(embed)
        s = nn.Sigmoid()
        sigmoid = s(pred)
        #log_softmax = F.log_softmax(pred, dim=0)
        return sigmoid, pred, embed

    def getWeight(self):
        return self.fcA1.weight.data.numpy(), self.fcB1.weight.data.numpy()

    def setWeight(self, a, b):
        self.fcA1.weight.data = torch.FloatTensor(a)
        self.fcB1.weight.data = torch.FloatTensor(b)

def enrichment(order, labels):
    bins = np.zeros(shape=(order.shape[1]))
    count = order.shape[0]
    for i in range(count):
        row = list(order[i])
        l = labels[i]
        posi = row.index(l)
        bins[posi] += 1

    for i in range(1, bins.shape[0]):
        bins[i] += bins[i-1]
    return sum(bins) / bins.shape[0] / count

def updateweight(target, train_feat, test_feat, train_ids, test_ids):
    reg = LinearRegression()
    train_target = target[train_ids]
    test_target = target[test_ids]
    reg.fit(train_feat, train_target)
    test_pred = reg.predict(test_feat)

    for i in range(len(test_ids)):
        row = test_ids[i]
        target[row] = test_pred[i]
    return target


if __name__ == "__main__":
    radius = 1
    epoch = 100
    train_repos, test_repos = loadnormaldata(radius)
    onehot_size = train_repos.onehot_size
    embed_size = 100
    
    model = Net(onehot_size, embed_size)
    

    train_ids = test_repos.input_ids
    test_ids = test_repos.output_ids
    #train_feat = test_repos.drug_feat[train_ids]
    #test_feat = test_repos.drug_feat[test_ids]
    reg = LinearRegression()
    
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss() # we can try other loss function later

    test_input, test_output_onehot, test_output_seq = test_repos.miniBatch(568)
    i = 0
    for i in range(epoch):
        while train_repos.Epoch:
            train_input, train_output_onehot, train_output_seq = train_repos.miniBatch(568)
            optimizer.zero_grad()
            sigmoid, pred, embed = model(train_input)
            #print("pred", pred.shape)
            #print("sig", sigmoid.shape)
            #print("train", train_output_onehot.shape)#, train_output_onehot.dtype)
            loss = criterion(pred.flatten(), train_output_onehot.flatten())
            loss.backward()
            optimizer.step()

        sigmoid, pred, embed = model(test_input)
        #loss = criterion(log_softmax, test_output_seq)

        out = sigmoid.data.numpy()
        labels = test_output_onehot.data.numpy()
        #order = np.flip(out.argsort(axis=1), axis=1)
        #print(labels.shape, pred.shape)
        auc = metrics.roc_auc_score(labels.flatten(), pred.data.numpy().flatten())
        print(i, auc)

        if i == epoch - 1:
            fpr, tpr, threshold = metrics.roc_curve(labels.flatten(), pred.data.numpy().flatten())
            plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % auc)
            plt.title('ROC')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.savefig('ROC.png')
        
        train_repos.reset()

        whole_input = torch.FloatTensor(np.identity(onehot_size))
        log_softmax, pred, embed = model(whole_input)
        embed = embed.data.numpy()
        if i%10 == 9:
            np.save(str(i) + ".npy", embed)
            print("saved!")
        
