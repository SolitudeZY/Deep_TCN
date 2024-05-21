import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from utils import data_generator, save
import time
import matplotlib.pyplot as plt
device = torch.device('cuda')


class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=28,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.Out2Class = nn.Linear(28, 20)

    def forward(self, input):
        output, hn = self.rnn(input, None)
        # print('hn,shape:{}'.format(hn.shape))
        outreshape = output[:, :, 0]
        # print(outreshape.shape)
        tmp = self.Out2Class(outreshape)
        # print(tmp.shape)
        return tmp


model = RNN()
model = model.to(device)
print(model)

model = model.train()

# img_transform = transforms.Compose([transforms.ToTensor(),
#                                     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
# dataset_train = datasets.MNIST(root='./data', transform=img_transform, train=True, download=True)
# dataset_test = datasets.MNIST(root='./data', transform=img_transform, train=False, download=True)
#
# train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=64, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=64, shuffle=False)


# images,label = next(iter(train_loader))
# print(images.shape)
# print(label.shape)
# images_example = torchvision.utils.make_grid(images)
# images_example = images_example.numpy().transpose(1,2,0)
# mean = [0.5,0.5,0.5]
# std = [0.5,0.5,0.5]
# images_example = images_example*std + mean
# plt.imshow(images_example)
# plt.show()

# 数据集位置
root = './dataset'

train_loader, test_loader = data_generator(root, 128)
# images,label = next(iter(train_loader))
# print(images.shape)
# print(label.shape)
# images_example = torchvision.utils.make_grid(images)
# images_example = images_example.numpy().transpose(1,2,0)
# mean = [0.5,0.5,0.5]
# std = [0.5,0.5,0.5]
# images_example = images_example*std + mean
# plt.imshow(images_example)
# plt.show()
def Get_ACC():
    # model = torch.load('rnn-model.pt')
    correct = 0
    test_loss = 0
    correct = 0
    total = 0
    classnum = 20
    target_num = torch.zeros((1, classnum))
    predict_num = torch.zeros((1, classnum))
    acc_num = torch.zeros((1, classnum))

    total_num = len(test_loader.dataset)
    startTest = time.perf_counter()
    for item in test_loader:
        batch_imgs, batch_labels = item
        batch_imgs = batch_imgs.squeeze(1)
        batch_imgs = Variable(batch_imgs)
        batch_imgs = batch_imgs.to(device)
        batch_labels = batch_labels.to(device)
        out = model(batch_imgs)
        _, pred = torch.max(out.data, 1)

        total += batch_labels.size(0)
        correct += pred.eq(batch_labels.data).cpu().sum()
        pre_mask = torch.zeros(out.size()).scatter_(1, pred.cpu().view(-1, 1), 1.)
        predict_num += pre_mask.sum(0)
        tar_mask = torch.zeros(out.size()).scatter_(1, batch_labels.data.cpu().view(-1, 1), 1.)
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
    correct = correct.data.item()
    acc = correct / total_num
    print('correct={},Test ACC:{:.5}'.format(correct, acc))


optimizer = torch.optim.Adam(model.parameters())
loss_f = nn.CrossEntropyLoss()

for epoch in range(3):
    print('epoch:{}'.format(epoch))
    cnt = 0
    startTrain = time.perf_counter()
    for item in train_loader:
        batch_imgs, batch_labels = item
        batch_imgs = batch_imgs.squeeze(1)
        # print(batch_imgs.shape)
        batch_imgs, batch_labels = Variable(batch_imgs), Variable(batch_labels)
        batch_imgs = batch_imgs.to(device)
        batch_labels = batch_labels.to(device)
        out = model(batch_imgs)
        # print(out.shape)
        loss = loss_f(out, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (cnt % 100 == 0):
            print_loss = loss.data.item()
            print('epoch:{},cnt:{},loss:{}'.format(epoch, cnt, print_loss))
        cnt += 1
    trainTime = (time.perf_counter() - startTrain)
    print("trainTime:", trainTime)
    Get_ACC()

save_filename = 'rnn-model.pt'
torch.save(model, save_filename)
print('Saved as %s' % save_filename)

# Get_ACC()
