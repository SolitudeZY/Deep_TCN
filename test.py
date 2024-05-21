import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from utils import data_generator
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
parser.add_argument('--epochs', type=int, default=5,
                    help='upper epoch limit (default: 5)')
# 卷积核大小
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
# 学习比率
parser.add_argument('--lr', type=float, default=2e-3,
                    help='initial learning rate (default: 2e-3)')
# 梯度下降算法（优化方法）
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
# 每层隐藏单元数
parser.add_argument('--nhid', type=int, default=25,
                    help='number of hidden units per layer (default: 25)')
# 随机数种子
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--permute', action='store_true',
                    help='use permuted MNIST (default: false)')
args = parser.parse_args()
input_channels = 1
seq_length = int(784 / input_channels)

# torch.Tensor是一种包含单一数据类型元素的多维矩阵。
# Permutation()函数的意思的打乱原来数据中元素的顺序
permute = torch.Tensor(np.random.permutation(784).astype(np.float64)).long()

# 数据集位置
root = './dataset'
train_loader, test_loader = data_generator(root, 128)

# 数据集分类
dict_20class = {0: 'BitTorrent', 1: 'Facetime', 2: 'FTP', 3: 'Gmail', 4: 'MySQL', 5: 'Outlook', 6: 'Skype', 7: 'SMB',
                8: 'Weibo', 9: 'WorldOfWarcraft', 10: 'Cridex', 11: 'Geodo', 12: 'Htbot', 13: 'Miuref', 14: 'Neris',
                15: 'Nsis-ay', 16: 'Shifu', 17: 'Tinba', 18: 'Virut', 19: 'Zeus'}

if __name__ == "__main__":
    model = torch.load('tcn-model.pt')
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    classnum = 20
    target_num = torch.zeros((1, classnum))
    predict_num = torch.zeros((1, classnum))
    acc_num = torch.zeros((1, classnum))
    # 是指停止自动求导
    # 是指停止自动求导
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data = data.view(-1, input_channels, seq_length)
            if args.permute:
                data = data[:, :, permute]
            # variable是tensor的外包装，data属性存储着tensor数据，grad属性存储关于该变量的导数，creator是代表该变量的创造者。
            data, target = Variable(data, volatile=True), Variable(target)
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
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
