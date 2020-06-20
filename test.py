from Dataset_process import *
from Model import *
import pandas as pd

voxel_test, seg_test, file_num_test = ReadData('data/test/candidate{}.npz', 584)

test_loader, test_size = \
    test_data_process(voxel_test, seg_test, 1)

net1 = DenseSharp()
net2 = DenseSharp()
net3 = DenseSharp()
# net4 = DenseSharp()
net1.load_state_dict(torch.load('model/net1.pkl'))
net2.load_state_dict(torch.load('model/net2.pkl'))
net3.load_state_dict(torch.load('model/net3.pkl'))
# net4.load_state_dict(torch.load('trained_net/test/1epoch_61_net1_trainloss_0.696_valloss_nan_accuracy_nan.pkl'))
net1.eval()
net2.eval()
net3.eval()
# net4.eval()
torch.cuda.empty_cache()
net1.cuda()
net2.cuda()
net3.cuda()
# net4.cuda()
output = torch.zeros(test_size) #sig
# output = torch.zeros(test_size,2) #sof
# mix_batch = 32
# for i in range(int(np.ceil(test_loader.dataset.shape[0]/mix_batch))):
#     if i == int(np.ceil(test_loader.dataset.shape[0]/mix_batch))-1:
#         test_loader.dataset[i*mix_batch:] = mixup_data_for_test(test_loader.dataset[i*mix_batch:])
#     else:
#         test_loader.dataset[i*mix_batch:(i+1)*mix_batch] \
#             = mixup_data_for_test(test_loader.dataset[i*mix_batch:(i+1)*mix_batch])
for j, voxel in enumerate(tqdm(test_loader)):
    voxel = voxel.cuda()
    prediction1 = net1(voxel).detach()
    prediction2 = net2(voxel).detach()
    prediction3 = net3(voxel).detach()
    # prediction4 = net4(voxel).detach()
    output[j] = torch.stack((prediction1, prediction2, prediction3), dim=0).mean(dim=0)
    # print(output[j])

rslt = pd.read_csv('data/sampleSubmission.csv')
# rslt['Predicted'] = output.numpy()[:,1] #sof
rslt['predicted'] = output.numpy() #sig
rslt.to_csv('submission.csv', index=False)