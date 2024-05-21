import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from utils import data_generator, save
from sklearn.metrics import classification_report
import time
from model import TCN
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.05,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit (default: 5)')
# 卷积核大小
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
# 学习比率
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 2e-3)')
# 梯度下降算法（优化方法）
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
# 每层隐藏单元数
parser.add_argument('--nhid', type=int, default=32,
                    help='number of hidden units per layer (default: 25)')
# 随机数种子
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--permute', action='store_true',
                    help='use permuted MNIST (default: false)')
args = parser.parse_args()

# 设计随机初始化种子，保证初始化都为固定
torch.manual_seed(args.seed)

# 判断显卡是否支持cuda
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# 数据集位置
root = './dataset'
# 每次扔进神经网络训练的数据个数
batch_size = args.batch_size
# TODO:
n_classes = 20
# TODO:
input_channels = 1
# TODO:
seq_length = int(784 / input_channels)
# 训练数据丢进神经网络次数
epochs = args.epochs
# TODO：预计是用于计算当前训练到的位置
steps = 0

# 输出所有参数
print(args)
# 预处理数据集
train_loader, test_loader = data_generator(root, batch_size)

# torch.Tensor是一种包含单一数据类型元素的多维矩阵。
# Permutation()函数的意思的打乱原来数据中元素的顺序
permute = torch.Tensor(np.random.permutation(784).astype(np.float64)).long()

# 设置通道数 = 隐藏层数*每层隐藏单元数
channel_sizes = [args.nhid] * args.levels
# 卷积核大小
kernel_size = args.ksize
# 生成模型
model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=args.dropout)

# 如果显卡支持cuda,那么将模型转化为cuda模型
if args.cuda:
    model.cuda()
    permute = permute.cuda()

# 设置损失函数、梯度和学习率
lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

def train(ep):
    global steps
    train_loss = 0
    # 在train模式下，dropout网络层会按照设定的参数p设置保留激活单元的概率（保留概率=p); batchnorm层会继续计算数据的mean和var等参数并更新
    model.train()
    # 使用 time.perf_counter() 作为替代
    startTrain = time.perf_counter()
    # startTrain = time.clock()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda: data, target = data.cuda(), target.cuda()
        # view是改变tensor的形状，view中的-1是自适应的调整
        data = data.view(-1, input_channels, seq_length)
        # 调换Tensor中各维度的顺序
        if args.permute:
            data = data[:, :, permute]
        data, target = Variable(data), Variable(target)
        # 变量梯度设置为0
        optimizer.zero_grad()
        output = model(data)
        # 计算损失数值
        loss = F.nll_loss(output, target)
        # 进行反向传播
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        # 进行梯度优化
        optimizer.step()
        # 计算损失值
        train_loss += loss
        # 计算当前到第多少个数据集
        steps += seq_length
        # 判断是否到达判断间隔组
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                ep, batch_idx * batch_size, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), train_loss.item() / args.log_interval, steps))
            train_loss = 0
    trainTime = (time.perf_counter() - startTrain)
    # trainTime = (time.clock() - startTrain)
    print("trainTime:", trainTime)


def test():
    # 在val模式下，dropout层会让所有的激活单元都通过，而batchnorm层会停止计算和更新mean和var，直接使用在训练阶段已经学出的mean和var值。
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    classnum = 20
    target_num = torch.zeros((1, classnum))
    predict_num = torch.zeros((1, classnum))
    acc_num = torch.zeros((1, classnum))
    startTest = time.perf_counter()
    # startTest = time.clock()
    # 是指停止自动求导
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data = data.view(-1, input_channels, seq_length)
            if args.permute:
                data = data[:, :, permute]
            # variable是tensor的外包装，data属性存储着tensor数据，grad属性存储关于该变量的导数，creator是代表该变量的创造者。
            with torch.no_grad():
                data, target = Variable(data), Variable(target)
            # data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # 计算正确率的操作
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            total += target.size(0)
            pre_mask = torch.zeros(output.size()).scatter_(1, pred.cpu().view(-1, 1), 1.)
            predict_num += pre_mask.sum(0)
            tar_mask = torch.zeros(output.size()).scatter_(1, target.data.cpu().view(-1, 1), 1.)
            target_num += tar_mask.sum(0)
            acc_mask = pre_mask * tar_mask
            acc_num += acc_mask.sum(0)
            test_loss /= len(test_loader.dataset)
            ###
            # 首先定义一个 分类数*分类数 的空混淆矩阵
            conf_matrix = torch.zeros(20, 20)
            # 使用torch.no_grad()可以显著降低测试用例的GPU占用
            conf_matrix = confusion_matrix(output, target, conf_matrix)
            conf_matrix = np.array(conf_matrix.cpu())  # 将混淆矩阵从gpu转到cpu再转到np
            corrects = conf_matrix.diagonal(offset=0)  # 抽取对角线的每种分类的识别正确个数
            per_kinds = conf_matrix.sum(axis=1)  # 抽取每个分类数据总的测试条数

            print("混淆矩阵总元素个数：{0},测试集总个数:{1}".format(int(np.sum(conf_matrix)), 123))
            print(conf_matrix)
            ###
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
        print('testSize{}'.format(len(test_loader.dataset)))
        print('recall', " ".join('%s' % id for id in recall))
        print('precision', " ".join('%s' % id for id in precision))
        print('F1', " ".join('%s' % id for id in F1))
        print('accuracy', accuracy)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        return test_loss


if __name__ == "__main__":
    for epoch in range(1, epochs + 1):
        train(epoch)
        test()
        save(model)
        if epoch % 10 == 0:
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
