import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms

from utils import data_generator


# 定义卷积神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3),  # b, 16(高度), 26, 26
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),  # b, 32, 24, 24
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # b, 32, 12, 12
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),  # b, 64, 10, 10
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),  # b, 128, 8, 8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # b, 128, 4, 4
        )

        self.out = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),  # (input, output)
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),  # (input, output)
            nn.ReLU(inplace=True),
            nn.Linear(128, 20)  # (input, output)
        )

    def forward(self, x):
        x = self.layer1(x)  # (batch, 16, 26, 26) -> (batchsize, 输出图片高度, 输出图片长度, 输出图片宽度)
        x = self.layer2(x)  # (batch, 32, 12, 12)
        x = self.layer3(x)  # (batch, 64, 10, 10)
        x = self.layer4(x)  # (batch, 128, 4, 4)
        x = x.view(x.size(0), -1)  # 扩展、展平 -> (batch, 128 * 4 * 4)
        x = self.out(x)
        return x

# 定义超参数
batch_size = 128
learning_rate = 1e-2

# 数据集位置
root = './dataset'

if __name__ == '__main__':
    # 数据预处理
    data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    # 下载训练集-MNIST手写数字训练集
    # train_dataset = datasets.MNIST(root="./data", train=True, transform=data_tf, download=True)
    # test_dataset = datasets.MNIST(root="./data", train=False, transform=data_tf)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # 预处理数据集
    train_loader, test_loader = data_generator(root, batch_size)
    model = CNN()
    if torch.cuda.is_available():
        model = model.cuda()
    # 定义损失函数和优化函数
    criterion = nn.CrossEntropyLoss()  # 损失函数：损失函数交叉熵
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # 优化函数：随机梯度下降法
    # 训练网络
    for epoch in range(5):
        cnt = 0
        startTrain = time.perf_counter()
        for data in train_loader:
            img, label = data
            img = Variable(img)
            if torch.cuda.is_available():
                img = Variable(img).cuda()
                label = Variable(label).cuda()
            else:
                img = Variable(img)
                label = Variable(label)
            # 前向传播
            out = model(img)
            loss = criterion(out, label)
            # 反向传播
            optimizer.zero_grad()  # 梯度归零
            loss.backward()
            optimizer.step()  # 更新参数
            cnt += 1
            if (cnt) % 200 == 0:
                print('*' * 10)
                print('epoch:{},cnt{} loss is {:.4f}'.format(epoch,cnt,loss.item()))
                print('loss is {:.4f}'.format(loss.item()))
        trainTime = (time.perf_counter() - startTrain)
        print("trainTime:", trainTime)
        # 测试网络
        model.eval()
        eval_loss = 0
        eval_acc = 0
        correct = 0
        total = 0
        classnum = 20
        target_num = torch.zeros((1, classnum))
        predict_num = torch.zeros((1, classnum))
        acc_num = torch.zeros((1, classnum))

        startTest = time.perf_counter()

        for data in test_loader:
            img, label = data
            # img = img.view(img.size(0), -1)
            img = Variable(img)
            if torch.cuda.is_available():
                img = Variable(img).cuda()
                label = Variable(label).cuda()
            else:
                img = Variable(img)
                label = Variable(label)
            out = model(img)
            loss = criterion(out, label)
            eval_loss += loss.item() * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            eval_acc += num_correct.item()

            total += label.size(0)
            correct += pred.eq(label.data).cpu().sum()
            pre_mask = torch.zeros(out.size()).scatter_(1, pred.cpu().view(-1, 1), 1.)
            predict_num += pre_mask.sum(0)
            tar_mask = torch.zeros(out.size()).scatter_(1, label.data.cpu().view(-1, 1), 1.)
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
        print('Test Loss:{:.6f}, Acc:{:.6f}'.format(eval_loss / (len(test_loader.dataset)),
                                                    eval_acc / (len(test_loader.dataset))))
        # 保存
        # save_filename = 'lstm-model.pt'
        # torch.save(model, save_filename)
        # print('Saved as %s' % save_filename)

