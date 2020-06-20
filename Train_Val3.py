import pandas as pd
import torch.optim as optim
from torch.autograd import Variable
from Dataset_process import *
from Model import *


def train(trainset,epoch,train_loader,optimizer,net,loss_fct,losses,accuracies,train_batch_size):
    for batch_idx, (data, label) in enumerate(tqdm(train_loader, desc='trainset{} epoch{}'.format(trainset,epoch))):
        data = data.cuda()
        label = label.cuda()
        data, label = Variable(data), Variable(label)
        optimizer.zero_grad()
        output = net(data)
        loss = loss_fct(output, label.to(dtype=torch.float32))
        loss.backward()
        optimizer.step()
        losses = np.append(losses, loss.item())
        output[output >= 0.5] = 1
        output[output < 0.5] = 0
        accuracy = (sum(output == label)).to(dtype=torch.float32) / train_batch_size
        # pred = np.argmax(output.detach().cpu(), axis=1)
        # target = np.argmax(label.cpu(), axis=1)
        # accuracy = (sum(pred == target)).to(dtype=torch.float32) / train_batch_size
        accuracies = np.append(accuracies, accuracy.cpu())
        # print("loss:{}, accuracy:{}".format(loss, accuracy))
    return losses, accuracies


def val(epoch,val_loader,net1,net2,net3,loss_fct,losses_val,accuracies_val,val_batch_size):
    for batch_idx, (data, label) in enumerate(tqdm(val_loader, desc='valset epoch{}'.format(epoch))):
        data = data.cuda()
        label = label.cuda()
        data, label = Variable(data), Variable(label)
        # data, labels_a, labels_b, lam = mixup_data(data, label)
        output1 = net1(data).detach()
        output2 = net2(data).detach()
        output3 = net3(data).detach()
        output_all = torch.stack((output1, output2, output3), dim=0).mean(dim=0)
        # loss = mixup_criterion(loss_fct, output, labels_a, labels_b, lam)
        loss = loss_fct(output_all, label.to(dtype=torch.float32))
        losses_val = np.append(losses_val, loss.item())
        output_all[output_all >= 0.5] = 1
        output_all[output_all < 0.5] = 0
        accuracy = (sum(output_all == label)).to(dtype=torch.float32) / val_batch_size
        # pred = np.argmax(output_all.cpu(), axis=1)
        # target = np.argmax(label.cpu(), axis=1)
        # accuracy = (sum(pred == target)).to(dtype=torch.float32) / val_batch_size
        accuracies_val = np.append(accuracies_val, accuracy.cpu())
        # print("loss:{}, accuracy:{}".format(loss, accuracy))
    return losses_val, accuracies_val


print("begin to read data!")
voxel_train_val, seg_train_val, file_num_train_val = ReadData('data/train_val/candidate{}.npz', 584)
# np.save("npy/voxel_train_val.npy", voxel_train_val)
# np.save("npy/seg_train_val.npy", seg_train_val)
# voxel_train_val = np.load("npy/voxel_train_val.npy")
# seg_train_val = np.load("npy/seg_train_val.npy")
# file_num_train_val = 465
# label = one_hot_label('data/train_val.csv')
label = pd.read_csv('data/train_val.csv').values[:, 1].astype(int)
print("finish reading data!")


train_batch_size = 16
val_batch_size = 16
LR = 3e-20
w_decay = 0.02
epoch = 50
min_loss_train = 1
min_loss_val = 1


print("begin to process data!")
train_voxel, train_mask, train_label, val_voxel, val_mask, val_label, \
    total_num_train, num_val = random_split(voxel_train_val, seg_train_val, label, ratio=0.05, train_set=3)
# np.save("npy/3new/train_voxel.npy", train_voxel)
# np.save("npy/3new/train_mask.npy", train_mask)
# np.save("npy/3new/train_label.npy", train_label)
# np.save("npy/3new/val_voxel.npy", val_voxel)
# np.save("npy/3new/val_mask.npy", val_mask)
# np.save("npy/3new/val_label.npy", val_label)
# train_voxel = np.load("npy/3new/train_voxel.npy")
# train_mask = np.load("npy/3new/train_mask.npy")
# train_label = np.load("npy/3new/train_label.npy")
# val_voxel = np.load("npy/3new/val_voxel.npy")
# val_mask = np.load("npy/3new/val_mask.npy")
# val_label = np.load("npy/3new/val_label.npy")
print("begin to load training data!")
train_loader1, train_size1 \
    = train_data_process(train_voxel[0], train_mask[0], train_label[0], train_batch_size)
print("trainloader1 finished!")
train_loader2, train_size2 \
    = train_data_process(train_voxel[1], train_mask[1], train_label[1], train_batch_size)
print("trainloader2 finished!")
train_loader3, train_size3 \
    = train_data_process(train_voxel[2], train_mask[2], train_label[2], train_batch_size)
print("trainloader3 finished!")
val_loader, val_size \
    = val_data_process(val_voxel, val_mask, val_label, val_batch_size,
                       his_equalized_after_crop=False)
print("valloader finished!")


print("begin to build models!")
torch.cuda.empty_cache()
net1 = DenseSharp()
optimizer1 = optim.Adam(net1.parameters(), lr=LR)
# optimizer1 = optim.SGD(net1.parameters(), momentum=0.9, lr=LR, weight_decay=w_decay)
net2 = DenseSharp()
optimizer2 = optim.Adam(net2.parameters(), lr=LR)
# optimizer2 = optim.SGD(net2.parameters(), momentum=0.9, lr=LR, weight_decay=w_decay)
net3 = DenseSharp()
optimizer3 = optim.Adam(net3.parameters(), lr=LR)
# optimizer3 = optim.SGD(net3.parameters(), momentum=0.9, lr=LR, weight_decay=w_decay)
loss_fct = nn.BCELoss()


net1.load_state_dict(torch.load('trained_net/net1.pkl'))
net2.load_state_dict(torch.load('trained_net/net2.pkl'))
net3.load_state_dict(torch.load('trained_net/net3.pkl'))


for i in range(epoch):
    net1.train()
    net2.train()
    net3.train()
    losses_train = []
    accuracies_train = []
    losses_val = []
    accuracies_val = []
    # if i >= 5:
    #     LR=3e-5*(1/np.floor(i/5+1))
    #     optimizer1 = optim.Adam(net1.parameters(), lr=LR)
    #     optimizer2 = optim.Adam(net2.parameters(), lr=LR)
    #     optimizer3 = optim.Adam(net3.parameters(), lr=LR)
        # optimizer1 = optim.SGD(net1.parameters(), momentum=0.9, lr=LR, weight_decay=w_decay)
        # optimizer2 = optim.SGD(net2.parameters(), momentum=0.9, lr=LR, weight_decay=w_decay)
        # optimizer3 = optim.SGD(net3.parameters(), momentum=0.9, lr=LR, weight_decay=w_decay)
    net1.cuda()
    losses_train, accuracies_train = \
        train(1,i+1,train_loader1,optimizer1,net1,loss_fct,losses_train,accuracies_train,train_batch_size)
    net1.cpu()
    net2.cuda()
    losses_train, accuracies_train = \
        train(2,i+1,train_loader2,optimizer2,net2,loss_fct,losses_train,accuracies_train,train_batch_size)
    net2.cpu()
    net3.cuda()
    losses_train, accuracies_train = \
        train(3,i+1,train_loader3,optimizer3,net3,loss_fct,losses_train,accuracies_train,train_batch_size)
    net3.cpu()
    avg_loss_train = np.mean(losses_train)
    avg_accuracy_train = np.mean(accuracies_train)
    print('avg_loss_train: ', avg_loss_train)
    print('avg_accuracy_train: ', avg_accuracy_train)
    net1.eval()
    net2.eval()
    net3.eval()
    torch.cuda.empty_cache()
    net1.cuda()
    net2.cuda()
    net3.cuda()
    losses_val, accuracies_val = \
        val(i+1, val_loader, net1, net2, net3, loss_fct, losses_val, accuracies_val, val_batch_size)
    net1.cpu()
    net2.cpu()
    net3.cpu()
    avg_loss_val = np.mean(losses_val)
    avg_accuracy_val = np.mean(accuracies_val)
    print('avg_loss_val: ', avg_loss_val)
    print('avg_accuracy_val: ', avg_accuracy_val)
    # if avg_loss_train < min_loss_train and avg_loss_val < 0.67:
    # if avg_loss_train < min_loss_train and avg_loss_val < min_loss_val:
    if avg_loss_val < min_loss_val:
    # if avg_loss_train < min_loss_train:
        min_loss_train = avg_loss_train
        min_loss_val = avg_loss_val
        net1_name = 'trained_net/epoch_' + str(i) + \
                    '_net1_trainloss_' + str(np.around(avg_loss_train, decimals=3)) + \
                    '_valloss_' + str(np.around(avg_loss_val, decimals=3)) + \
                    '_accuracy_' + str(np.around(avg_accuracy_val, decimals=3)) + \
                    '.pkl'
        net2_name = 'trained_net/epoch_' + str(i) + '_net2.pkl'
        net3_name = 'trained_net/epoch_' + str(i) + '_net3.pkl'
        torch.save(net1.state_dict(), net1_name)
        torch.save(net2.state_dict(), net2_name)
        torch.save(net3.state_dict(), net3_name)
        print('new net saved!')
    print("min_loss_train: ", min_loss_train)
    print("min_loss_val: ", min_loss_val)