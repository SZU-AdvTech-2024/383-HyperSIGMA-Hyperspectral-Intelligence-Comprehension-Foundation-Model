from model import ss_fusion_seg
import torch
from torch  import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report,cohen_kappa_score
from model import split_data,utils,create_graph
from sklearn import metrics, preprocessing
from mmengine.optim import build_optim_wrapper
from mmcv_custom import custom_layer_decay_optimizer_constructor,layer_decay_optimizer_constructor_vit
import random
import os
import torch.utils.data as Data
import copy
import scipy.io as sio
import spectral as spy
from collections import Counter
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
class PaviaURaw(DataReader):
    def __init__(self):
        super(PaviaURaw, self).__init__()

        raw_data_package = sio.loadmat(r"/home7/wsq/HyperSIGMA/ImageClassification/data/PaviaU.mat")
        self.data_cube = raw_data_package["paviaU"].astype(np.float32)
        truth = sio.loadmat(r"/home7/wsq/HyperSIGMA/ImageClassification/data/PaviaU_gt.mat")
        self.g_truth = truth["paviaU_gt"].astype(np.float32)

def load_data():
    data = PaviaURaw().cube
    data_gt =PaviaURaw().truth
    return data, data_gt

patch_size =8
img_size = 128
pca_components = 20
split_type = ['number', 'ratio'][1]
train_num = 10
val_num =0
train_ratio = 0.8
val_ratio = 0.1
max_epoch = 300
batch_size = 64
dataset_name = 'HH'

path_weight = r"/home7/wsq/HyperSIGMA/ImageClassification/seg_weights/"
path_result = r"/home7/wsq/HyperSIGMA/ImageClassification/seg_result/"
data, data_gt = load_data()
height_orgin, width_orgin, bands = data.shape
class_num = np.max(data_gt)
class_num = class_num.astype(int)

data, pca = split_data.apply_PCA(data, num_components=pca_components)
gt_reshape = np.reshape(data_gt, [-1])


train_index, val_index, test_index = split_data.split_data(gt_reshape,
            class_num, train_ratio, val_ratio, train_num, val_num, split_type)


#data_all, data_label_all = split_data.create_patches(data, data_gt, window_size=img_size, remove_zero_labels = False)
#

train_index=train_index.astype(int)
val_index=val_index.astype(int)
test_index=test_index.astype(int)



#根据输入的索引，将地面真实标签 (gt_reshape) 分别生成训练、验证和测试集的地面真实标签掩码。train_samples_gt、test_samples_gt、val_samples_gt：训练集、测试集和验证集的地面真实标签矩阵。
#train_samples_gt、test_samples_gt、val_samples_gt：分别为训练集、测试集和验证集的地面真实标签矩阵，只有相应索引位置有值
train_samples_gt, test_samples_gt, val_samples_gt = create_graph.get_label(gt_reshape,
                                                train_index, val_index, test_index)
#为训练集、测试集和验证集分别生成标签掩码，掩码矩阵用于标识哪些像素属于某个类别。生成掩码：对于每个像素，如果它属于某个类别（即标签不为 0），则该像素的掩码置为 1。
#独热形式：生成的掩码为独热编码形式，每个像素点的长度等于类别数。



train_gt = np.reshape(train_samples_gt,[height_orgin,width_orgin])
test_gt = np.reshape(test_samples_gt,[height_orgin,width_orgin])
val_gt = np.reshape(val_samples_gt,[height_orgin,width_orgin])

#将地面真实标签矩阵转换为独热编码形式。#train_label_mask、test_label_mask、val_label_mask：分别为训练集、测试集和验证集的标签掩码矩阵，形状为 [height * width, class_num]
train_gt_onehot = create_graph.label_to_one_hot(train_gt, class_num)
test_gt_onehot = create_graph.label_to_one_hot(test_gt, class_num)
val_gt_onehot = create_graph.label_to_one_hot(val_gt, class_num)


train_samples_gt=torch.from_numpy(train_samples_gt.astype(np.float32)).to(device)
test_samples_gt=torch.from_numpy(test_samples_gt.astype(np.float32)).to(device)
val_samples_gt=torch.from_numpy(val_samples_gt.astype(np.float32)).to(device)
train_gt_onehot = torch.from_numpy(train_gt_onehot.astype(np.float32)).to(device)
test_gt_onehot = torch.from_numpy(test_gt_onehot.astype(np.float32)).to(device)
val_gt_onehot = torch.from_numpy(val_gt_onehot.astype(np.float32)).to(device)


#返回图像的补丁列表 (sub_imgs)、行数 (num_H)、列数 (num_W)、更新后的地面真实标签 (img_gt)、以及填充后的图像 (img
#将输入图像和地面真实标签划分为多个补丁，适应模型的输入大小
#返回值：包含补丁信息、行列数量以及标签图，用于后续训练和测试的数据准备
img_train, num_H, num_W,data_gt,data = utils.Get_train_and_test_data(img_size, data,data_gt)
height, width, bands = data.shape
img_train = torch.from_numpy(img_train.transpose(0,3,1,2)).type(torch.FloatTensor)

# 计算每个子集的数量
#total_samples = img_train.shape[0]  # 第一维度的大小，即样本数
#train_size = int(total_samples * train_ratio)
#val_size = int(total_samples * val_ratio)
#test_size = total_samples - train_size - val_size  # 剩余的就是测试集大小

#indices = np.arange(total_samples)
# 根据比例获取训练、验证和测试集的索引
#train_indices = indices[:train_size]
#val_indices = indices[train_size:train_size + val_size]
#test_indices = indices[train_size + val_size:]

# 使用索引划分数据
#train_data = img_train[train_indices]
#val_data = img_train[val_indices]
#test_data = img_train[test_indices]

# 输出各子集的形状
#print("Train data shape:", train_data.shape)  # (12, 20, 128, 128)
#print("Validation data shape:", val_data.shape)  # (1, 20, 128, 128)
#print("Test data shape:", test_data.shape)  # (2, 20, 128, 128)





#使用 Data.TensorDataset() 创建了一个 PyTorch 张量数据集 data_train，其中 img_train 是作为训练数据的张量,img_train：这是之前生成的高光谱数据的补丁
#可以与 PyTorch 的 DataLoader 一起使用，便于后续的批量加载.
data_train = Data.TensorDataset(img_train)
#创建一个用于加载训练数据的 DataLoader,train_loader 将用于训练过程中的数据迭代.batch_size=num_H：每次从数据集中取出 num_H 个样本作为一个批次，num_H 是图像高度被补丁大小整除的行数
train_loader = Data.DataLoader(data_train, batch_size=num_H,shuffle=False)
#
val_loader = Data.DataLoader(data_train, batch_size=num_H,shuffle=False)
test_loader = Data.DataLoader(data_train, batch_size=num_H,shuffle=False)

train_samples_gt[train_index] = train_samples_gt[train_index] -1
zeros = torch.zeros([height_orgin * width_orgin]).to(device).float()

model = ss_fusion_seg.SSFusionFramework(
                img_size = img_size,
                in_channels = pca_components,
                patch_size=patch_size,
                classes = class_num,
                model_size='base'#The optional values are 'base','large' and 'huge'
)


model_params =model.state_dict()
spat_net = torch.load((r"/home7/wsq/HyperSIGMA/spat-base.pth"), map_location=torch.device('cpu'))
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
per_net = torch.load((r"/home7/wsq/HyperSIGMA/spec-base.pth"), map_location=torch.device('cpu'))
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
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizer, max_epoch, eta_min=0, last_epoch=-1)
criterion = nn.CrossEntropyLoss()
model.to(device)
count = 0
best_loss = 99999
train_losses = []
val_losses = []

for epoch in range(max_epoch+ 1):
    pred = torch.zeros([num_W, num_H, class_num, img_size, img_size])
    optimizer.zero_grad()
    for batch_idx, (batch_data) in enumerate(train_loader):
        for i in range(num_H):
            netinput = batch_data[0][i]
            netinput = torch.unsqueeze(netinput, 0).to(device)
            batch_pred = model(netinput)
            #batch_pred = batch_pred.detach()
            batch_pred = batch_pred.reshape(img_size,img_size,-1)
            batch_pred =batch_pred. permute(([2, 0, 1]), 0)
            pred[batch_idx,i] = batch_pred
    pred = torch.reshape(pred, [num_H, num_W, class_num, img_size, img_size])
    pred = torch.permute(pred, [2, 0, 3, 1, 4])  # [2,num_H, img_size,num_W, img_size]]
    pred = torch.reshape(pred, [class_num, num_H * img_size* num_W * img_size])
    pred = torch.permute(pred, [1, 0])
    y =pred.to(device)
    train_index =train_index.reshape(-1,)
    y_orgin =  utils.image_reshape(y,height,width,height_orgin,width_orgin,class_num)
    loss = criterion(y_orgin[train_index], train_samples_gt[train_index].long())
    loss.backward(retain_graph=False)
    optimizer.step()
    if epoch%10==0:
        trainOA = utils.evaluate_performance(y_orgin, train_samples_gt, train_gt_onehot, zeros)
        print("{}\ttrain loss={:.4f}\t train OA={:.4f} ".format(str(epoch + 1), loss, trainOA))
        if loss < best_loss :
            best_loss = loss
            print('save model')
            torch.save(model.state_dict(), path_weight + r"model.pt")

torch.cuda.empty_cache()
with torch.no_grad():
    model.load_state_dict(torch.load(path_weight + r"model.pt"))
    model.eval()
    pred = torch.zeros([num_W, num_H, class_num, img_size, img_size])
    for batch_idx, (batch_data) in enumerate(test_loader):
        for w in range(num_H):
            netinput = batch_data[0][w]
            netinput = torch.unsqueeze(netinput, 0).to(device)
            batch_pred = model(netinput)
            #batch_pred = batch_pred.detach()
            batch_pred = batch_pred.reshape(img_size,img_size,-1)
            batch_pred =batch_pred. permute(([2, 0, 1]), 0)
            pred[batch_idx,w] = batch_pred
    pred = torch.reshape(pred, [num_H, num_W, class_num, img_size, img_size])
    pred = torch.permute(pred, [2, 0, 3, 1, 4])  # [2,num_H, img_size,num_W, img_size]]
    pred = torch.reshape(pred, [class_num, num_H * img_size* num_W * img_size])
    pred = torch.permute(pred, [1, 0])
    y =pred.to(device)



    y_orgin = utils.image_reshape(y,height,width,height_orgin,width_orgin,class_num)


    overall_acc,OA_hi1,average_acc,kappa,each_acc=utils.evaluate_performance_all(y_orgin, test_samples_gt, test_gt_onehot,  height_orgin, width_orgin, class_num, test_gt,device, require_AA_KPP=True, printFlag=False)
    print("test OA={:.4f}".format(overall_acc))
    print('kappa=',kappa)
    print('each_acc=',each_acc)
    print('average_acc=',average_acc)

    y_orgin= y_orgin.cpu().numpy()  # 先将 GPU 张量移到 CPU
    y_orgin = np.argmax(y_orgin, axis=1)  # 使用 numpy 进行操作
    # 假设 y 的形状为 (65536, 1)
    y_orgin = y_orgin.reshape(height_orgin,width_orgin, -1)  # 重塑为 (256, 256, 1)

    # 如果需要将最后一个维度调整为动态的，可以保持其不固定为 1


def imshow_IP(data,class_num,height_orgin,width_orgin,name):
    colormap = np.zeros((10, 3))
    colormap[1, :] = [255/255, 0/255, 0/255]
    colormap[2, :] = [0, 255/255, 0]
    colormap[3, :] = [0, 0/255, 255/255]
    colormap[4, :] = [255/255, 255/255, 0]
    colormap[5, :] = [0/255, 255/255, 255/255]
    colormap[6, :] = [255/255, 0/255, 255/255]
    colormap[7, :] = [176/255, 48/255, 96/255]
    colormap[8, :] = [46/255, 139/255, 87/255]
    colormap[9, :] = [160/255, 32/255, 240/255]

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

#imshow_IP(y_orgin,class_num,height,width,r"/home7/wsq/HyperSIGMA/ImageClassification/seg_result/classification_result.png")