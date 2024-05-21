# -*- coding: utf-8 -*-
import torch
from torch import nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from utils import data_generator, save
import time
from torch.utils.data import DataLoader

# torch.manual_seed(1)    # reproducible


num_epoches = 3
BATCH_SIZE = 128  # 批训练的数量
TIME_STEP = 28  # 相当于序列长度
INPUT_SIZE = 28  # 特征向量长度
LR = 0.001  # learning rate

# MNIST数据集下载
# train_dataset = datasets.MNIST(
#     root='./data', train=True, transform=transforms.ToTensor(), download=True)
#
# test_dataset = datasets.MNIST(
#     root='./data', train=False, transform=transforms.ToTensor())
#
# train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 数据集位置
root = './dataset'

train_loader, test_loader = data_generator(root, BATCH_SIZE)


# 定义网络模型
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(input_size=INPUT_SIZE,  # if use nn.RNN(), it hardly learns
                           hidden_size=64,  # rnn 隐藏单元
                           num_layers=1,  # rnn 层数
                           batch_first=True,
                           # input & output will have batch size as 1s dimension. e.g. (batch, seq, input_size)
                           )
        self.out = nn.Linear(64, 20)  # 10分类

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # None represents zero initial hidden state
        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


rnn = RNN()  # 实例化
rnn = rnn.cuda()
print(rnn)  # 查看模型结构

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # 选择优化器，optimize all cnn parameters
criterion = nn.CrossEntropyLoss()  # 定义损失函数，the target label is not one-hotted

for epoch in range(num_epoches):
    print('epoch {}'.format(epoch + 1))
    print('*' * 10)
    running_loss = 0.0
    running_acc = 0.0
    # time.clock() 在3.3后被弃置
    startTrain = time.perf_counter()
    rnn.train()
    for imgs, labels in train_loader:
        imgs = imgs.squeeze(1)  # (N,28,28)
        imgs = Variable(imgs.cuda())
        labels = Variable(labels.cuda())
        # 前向传播
        out = rnn(imgs)
        loss = criterion(out, labels)
        running_loss += loss.item() * labels.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == labels).sum()
        running_acc += num_correct.item()
        # 向后传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    trainTime = (time.perf_counter() - startTrain)
    print("trainTime:", trainTime)

    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, running_loss / (len(train_loader.dataset)), running_acc / (len(train_loader.dataset))))

rnn.eval()
eval_loss = 0.0
eval_acc = 0.0
correct = 0
total = 0
classnum = 20
target_num = torch.zeros((1, classnum))
predict_num = torch.zeros((1, classnum))
acc_num = torch.zeros((1, classnum))
startTest = time.perf_counter()
for imgs, labels in test_loader:
    imgs = imgs.squeeze(1)  # (N,28,28)
    imgs = imgs.cuda()
    labels = labels.cuda()

    out = rnn(imgs)
    loss = criterion(out, labels)
    eval_loss += loss.item() * labels.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == labels).sum()
    eval_acc += num_correct.item()

    total += labels.size(0)
    correct += pred.eq(labels.data).cpu().sum()
    pre_mask = torch.zeros(out.size()).scatter_(1, pred.cpu().view(-1, 1), 1.)
    predict_num += pre_mask.sum(0)
    tar_mask = torch.zeros(out.size()).scatter_(1, labels.data.cpu().view(-1, 1), 1.)
    target_num += tar_mask.sum(0)
    acc_mask = pre_mask * tar_mask
    acc_num += acc_mask.sum(0)
testTime = (time.perf_counter() - startTest)
print("testTime:", testTime)
recall = acc_num / target_num
precision = acc_num / predict_num
F1 = 2 * recall * precision / (recall + precision)
accuracy = acc_num.sum(1) / target_num.sum(1)
# 精度调整
recall = (recall.numpy()[0] * 100).round(3)
precision = (precision.numpy()[0] * 100).round(3)
F1 = (F1.numpy()[0] * 100).round(3)
accuracy = (accuracy.numpy()[0] * 100).round(3)
# 打印格式方便复制
print('recall', " ".join('%s' % id for id in recall))
print('precision', " ".join('%s' % id for id in precision))
print('F1', " ".join('%s' % id for id in F1))
print('accuracy', accuracy)
print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
    test_loader.dataset)), eval_acc / (len(test_loader.dataset))))
# save_filename = 'lstm-model.pt'
# torch.save(rnn, save_filename)
# print('Saved as %s' % save_filename)

# rnn = torch.load('lstm-model.pt')
# rnn.eval()
# eval_loss = 0.0
# eval_acc = 0.0
# correct = 0
# total = 0
# classnum = 20
# target_num = torch.zeros((1, classnum))
# predict_num = torch.zeros((1, classnum))
# acc_num = torch.zeros((1, classnum))
# for imgs, labels in test_loader:
#     imgs = imgs.squeeze(1)  # (N,28,28)
#     imgs = imgs.cuda()
#     labels = labels.cuda()
#
#     out = rnn(imgs)
#     loss = criterion(out, labels)
#     eval_loss += loss.item() * labels.size(0)
#     _, pred = torch.max(out, 1)
#     num_correct = (pred == labels).sum()
#     eval_acc += num_correct.item()
#
#     total += labels.size(0)
#     correct += pred.eq(labels.data).cpu().sum()
#     pre_mask = torch.zeros(out.size()).scatter_(1, pred.cpu().view(-1, 1), 1.)
#     predict_num += pre_mask.sum(0)
#     tar_mask = torch.zeros(out.size()).scatter_(1, labels.data.cpu().view(-1, 1), 1.)
#     target_num += tar_mask.sum(0)
#     acc_mask = pre_mask * tar_mask
#     acc_num += acc_mask.sum(0)
# recall = acc_num / target_num
# precision = acc_num / predict_num
# F1 = 2 * recall * precision / (recall + precision)
# accuracy = acc_num.sum(1) / target_num.sum(1)
# # 精度调整
# recall = (recall.numpy()[0] * 100).round(3)
# precision = (precision.numpy()[0] * 100).round(3)
# F1 = (F1.numpy()[0] * 100).round(3)
# accuracy = (accuracy.numpy()[0] * 100).round(3)
# # 打印格式方便复制
# print('recall', " ".join('%s' % id for id in recall))
# print('precision', " ".join('%s' % id for id in precision))
# print('F1', " ".join('%s' % id for id in F1))
# print('accuracy', accuracy)
# print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
#     test_loader.dataset)), eval_acc / (len(test_loader.dataset))))
