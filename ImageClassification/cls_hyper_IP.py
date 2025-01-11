from model import ss_fusion_cls
import torch
from torch  import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report,cohen_kappa_score
from model import split_data,utils
from sklearn import metrics, preprocessing
from mmengine.optim import build_optim_wrapper
from mmcv_custom import custom_layer_decay_optimizer_constructor,layer_decay_optimizer_constructor_vit
import scipy.io as sio
from thop import profile
from multiprocessing import shared_memory
from matplotlib.colors import ListedColormap

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda")
print(device)

class DataReader():
    def __init__(self):
        self.data_cube = None
        self.g_truth = None

    @property
    def cube(self):
        """
        origin data
        """
        return self.data_cube

    @property
    def truth(self):
        return self.g_truth

    @property
    def normal_cube(self):
        """
        normalization data: range(0, 1)
        """
        return (self.data_cube - np.min(self.data_cube)) / (np.max(self.data_cube) - np.min(self.data_cube))
class IndianRaw(DataReader):
    def __init__(self):
        super(IndianRaw, self).__init__()
        raw_data_package = sio.loadmat(r"data/Indian_pines_corrected.mat")
        self.data_cube = raw_data_package["data"].astype(np.float32)
        truth = sio.loadmat(r"data/Indian_pines_gt.mat")
        self.g_truth = truth["groundT"].astype(np.float32)

def load_data():
    data = IndianRaw().cube
    data_gt = IndianRaw().truth
    return data, data_gt

img_size =33
patch_size = 2
pca_components = 30
split_type = ['number', 'ratio'][1]
train_num =10
val_num =5
train_ratio = 0.8
val_ratio = 0.1

max_epoch = 1
batch_size = 16
learning_rate = 0.00001
dataset_name = 'HH'

path_weight = r"/home5/wsq/HyperSIGMA/ImageClassification/weights/"
path_result = r"/home5/wsq/HyperSIGMA/ImageClassification/result/"
data, data_gt = load_data()
height, width, bands = data.shape
gt_reshape = np.reshape(data_gt, [-1])
class_num = np.max(data_gt)
class_num = class_num.astype(int)

data, data_gt = load_data()
data, pca = split_data.apply_PCA(data, num_components=pca_components)
train_index, val_index, test_index = split_data.split_data(gt_reshape,
            class_num, train_ratio, val_ratio, train_num, val_num, split_type)


data_all, data_label_all = split_data.create_patches(data, data_gt, window_size=img_size, remove_zero_labels = False)

train_index=train_index.astype(int)
val_index=val_index.astype(int)
test_index=test_index.astype(int)
height, width, bands = data.shape
gt_reshape = np.reshape(data_gt, [-1])
class_num = np.max(gt_reshape)
class_num = class_num.astype(int)

train_index0, val_index0, re_index = split_data.split_data(gt_reshape,
            class_num, 0, 0, 0, 10,'ratio' )
re_index = np.sort(re_index)

train_index = train_index.reshape(-1,1)
val_index = val_index.reshape(-1,1)
test_index = test_index.reshape(-1,1)

train_label = data_label_all[train_index].reshape(-1,)
val_label = data_label_all[val_index].reshape(-1,)
test_label = data_label_all[test_index]

train = data_all[train_index,:,:,:]
val = data_all[val_index,:,:,:]
test = data_all[test_index,:,:,:]

split_data.data_info(train_label, val_label, test_label, np.max(data_label_all))

train = train.reshape(-1, img_size, img_size, bands)
val = val.reshape(-1, img_size, img_size, bands)
test = test.reshape(-1,img_size, img_size, bands)
print('before transpose: train shape: ', train.shape)
print('before transpose: test  shape: ', val.shape)
print('before transpose: test  shape: ', test.shape)

train = train.transpose(0, 3, 1, 2)
val = val.transpose(0,  3, 1,2 )
test = test.transpose(0,  3, 1,2 )
print('after transpose: train shape: ', train.shape)
print('after transpose: val  shape: ', val.shape)
print('after transpose: test  shape: ', test.shape)

train_hi_gt= torch.zeros(train_label.size,class_num)
val_hi_gt= torch.zeros(val_label.size,class_num)
test_hi_gt= torch.zeros(test_label.size,class_num)

print(train_hi_gt.shape)
print(val_hi_gt.shape)
print(test_hi_gt.shape)


train_hi_gt = torch.LongTensor(train_hi_gt.numpy())
test_hi_gt = torch.LongTensor(test_hi_gt.numpy())
val_hi_gt = torch.LongTensor(val_hi_gt.numpy())

class TrainDS(torch.utils.data.Dataset):
    def __init__(self):
        self.len = train.shape[0]
        self.x_data = torch.FloatTensor(train)
        self.y_data = torch.LongTensor(train_label)
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


""" Val dataset"""


class ValDS(torch.utils.data.Dataset):
    def __init__(self):
        self.len = val.shape[0]
        self.x_data = torch.FloatTensor(val)
        self.y_data = torch.LongTensor(val_label)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


""" Testing dataset"""


class TestDS(torch.utils.data.Dataset):
    def __init__(self):
        self.len = test.shape[0]
        self.x_data = torch.FloatTensor(test)
        self.y_data = torch.LongTensor(test_label)
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


# 创建 trainloader 和 testloader
trainset = TrainDS()
valset = ValDS()
testset = TestDS()
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=64, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(dataset=valset, batch_size=64, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=64, shuffle=False, num_workers=0)


model = ss_fusion_cls.SSFusionFramework(
                img_size = img_size,
                in_channels = pca_components,
                patch_size=patch_size,
                classes = class_num+1,
                model_size='base' #The optional values are 'base','large' and 'huge'
)

model_params =model.state_dict()
spat_net = torch.load((r"/home5/wsq/HyperSIGMA/spat-base.pth"), map_location=torch.device('cuda'))
for k in list(spat_net['model'].keys()):
    if 'patch_embed.proj' in k:
        del spat_net['model'][k]
for k in list(spat_net['model'].keys()):
    if 'spat_map' in k:
        del spat_net['model'][k]
for k in list(spat_net['model'].keys()):
    if 'spat_output_maps' in k:
        del spat_net['model'][k]
for k in list(spat_net['model'].keys()):
    if 'pos_embed' in k:
        del spat_net['model'][k]
spat_weights = {}
prefix = 'spat_encoder.'
for key, value in spat_net['model'].items():
    new_key = prefix + key
    spat_weights[new_key] = value
per_net = torch.load((r"/home5/wsq/HyperSIGMA/spec-base.pth"), map_location=torch.device('cuda'))
model_params =model.state_dict()
for k in list(per_net['model'].keys()):
    if 'patch_embed.proj' in k:
        del per_net['model'][k]
    if 'spat_map' in k:
        del per_net['model'][k]
    if 'fpn1.0.weight' in k:
        del per_net['model'][k]
spec_weights = {}
prefix = 'spec_encoder.'
for key, value in per_net['model'].items():
    new_key = prefix + key
    spec_weights[new_key] = value
model_params =model.state_dict()
for k in list(spec_weights.keys()):
    if 'spec_encoder.patch_embed' in k:
        del spec_weights[k]
merged_params = {**spat_weights, **spec_weights}
same_parsms = {k: v for k, v in merged_params.items() if k in model_params.keys()}
model_params.update(same_parsms)
model.load_state_dict(model_params)


optim_wrapper = dict(
    optimizer=dict(
    type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.05),
    constructor='LayerDecayOptimizerConstructor_ViT',
    paramwise_cfg=dict(
        num_layers=12,
        layer_decay_rate=0.9,
        )
        )
optimizer = build_optim_wrapper(model, optim_wrapper)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizer, 300, eta_min=0, last_epoch=-1)
criterion = nn.CrossEntropyLoss()
model.to(device)
count = 0
best_loss = 99999
train_losses = []
val_losses = []

for epoch in range(max_epoch + 1):
    correct = 0
    total = 0
    test_correct = 0
    test_total = 0
    _train_loss = 0
    val_correct = 0
    model.train()
    for x, y in train_loader:
        if torch.cuda.is_available():
            x, y = x.to(device), y.to(device)
        y_pred = model(x)
        train_loss = criterion(y_pred, y)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        _train_loss += train_loss.item()

        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            correct = (y_pred == y).sum().item()
            total = y.size(0)

    train_losses.append(train_loss.cpu().detach().item())

    if epoch % 10 == 0:

        epoch_loss = _train_loss / len(train_loader.dataset)
        epoch_accuracy = correct / total
        if epoch_loss < best_loss:
            best_loss = epoch_loss

            # 检查并创建目录
            # os.makedirs(path_weight, exist_ok=True)

            torch.save(model.state_dict(), path_weight + "model.pth")
            torch.save(optimizer.state_dict(), path_weight + 'optimizer.pth')
            print('save model')

        print('epoch', epoch,
              'loss: ', round(epoch_loss, 5),
              'accuracy: ', round(epoch_accuracy, 5),
              )


count = 0
model.load_state_dict(torch.load(path_weight + r"model.pth"))
model.eval()
with torch.no_grad():
    for x, y  in test_loader:
        if torch.cuda.is_available():
            x, y = x.to(device), y.to(device)
        output = model(x)
        y_pred = torch.argmax(output, dim=1)
        test_correct += (y_pred == y).sum().item()
        test_total += y.size(0)

        if count == 0:
            y_pred_test =  y_pred.cpu().numpy()
            y_gt_test = y.cpu().numpy()
            count = 1
        else:
            y_pred_test = np.concatenate( (y_pred_test, y_pred.cpu().numpy()) )
            y_gt_test = np.concatenate( (y_gt_test, y.cpu().numpy()) )

overall_acc = metrics.accuracy_score(y_pred_test, y_gt_test)
confusion_matrix = metrics.confusion_matrix(y_pred_test, y_gt_test)
each_acc, average_acc = utils.aa_and_each_accuracy(confusion_matrix)
kappa = metrics.cohen_kappa_score(y_pred_test, y_gt_test)
OA_HybirdSN = test_correct / test_total
print("test OA={:.4f}".format(overall_acc))
print('kappa=',kappa)
print('each_acc=',each_acc)
print('average_acc=',average_acc)

print(y_pred_test.shape)
print(y_gt_test.shape)



def ConfusionMatrix(output,label,index,class_num):
    conf_matrix = np.zeros((class_num+1,class_num+1)).astype(int)
    preds = torch.argmax(output, 1)
    preds=np.array(preds.cpu()).astype(int)
    preds = preds
    label=np.array(label.cpu()).astype(int)
    label = label
    for p, t in zip(preds[index], label[index]):
        conf_matrix[p, t] += 1
    return conf_matrix

matrix=confusion_matrix(y_pred_test,y_gt_test,class_num)

def imshow_IP(data,class_num,height_orgin,width_orgin,name):
    colormap = np.zeros((17, 3))
    colormap[1, :] = [255/255, 0/255, 0/255]
    colormap[2, :] = [0, 255/255, 0]
    colormap[3, :] = [0, 0/255, 255/255]
    colormap[4, :] = [255/255, 255/255, 0]
    colormap[5, :] = [0/255, 255/255, 255/255]
    colormap[6, :] = [255/255, 0/255, 255/255]
    colormap[7, :] = [176/255, 48/255, 96/255]
    colormap[8, :] = [46/255, 139/255, 87/255]
    colormap[9, :] = [160/255, 32/255, 240/255]
    colormap[10, :] = [255/255, 127/255, 80/255]
    colormap[11, :] = [127/255, 255/255, 212/255]
    colormap[12, :] = [218/255, 112/255, 214/255]
    colormap[13, :] = [160/255, 82/255, 45/255]
    colormap[14, :] = [127/255, 255/255, 0/255]
    colormap[15, :] = [216/255, 191/255, 216/255]
    colormap[16, :] = [238/255, 0/255, 0/255]

    #h,w = data.shape
    truthmap = np.zeros((height_orgin,width_orgin, 3), dtype=np.float32)
    for k in range(1, class_num + 1):
        for i in range(height_orgin):
            for j in range(width_orgin):
                if data[i, j] == k:
                    truthmap[i, j, :] = colormap[k, :]
    plt.figure()
    plt.imshow(truthmap)
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(name,dpi=360)
    plt.show()

#imshow_IP(y_orgin,class_num,height_orgin,width_orgin,r"/home7/wsq/HyperSIGMA/ImageClassification/seg_result/classification_result.png")


#imshow_IP(data_gt,class_num,height_orgin,width_orgin,r"/home7/wsq/HyperSIGMA/ImageClassification/seg_result/classification_result.png")











