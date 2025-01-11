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
from sklearn.metrics import confusion_matrix
import seaborn as sns

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
class PaviaURaw(DataReader):
    def __init__(self):
        super(PaviaURaw, self).__init__()

        raw_data_package = sio.loadmat(r"data/PaviaU.mat")
        self.data_cube = raw_data_package["paviaU"].astype(np.float32)
        truth = sio.loadmat(r"data/PaviaU_gt.mat")
        self.g_truth = truth["paviaU_gt"].astype(np.float32)
class HongHuRaw(DataReader):
    def __init__(self):
        super(HongHuRaw, self).__init__()
        raw_data_package = sio.loadmat(r"data/HongHu.mat")
        #print(raw_data_package)
        self.data_cube = raw_data_package["WHU_Hi_HongHu"].astype(np.float32)
        truth = sio.loadmat(r"data/HongHu_gt.mat")
        #print(truth)
        self.g_truth = truth["WHU_Hi_HongHu_gt"].astype(np.float32)
class Houston2013trRaw(DataReader):
    def __init__(self):
        super(Houston2013trRaw, self).__init__()

        raw_data_package = sio.loadmat(r"data/Houston.mat")
        print(raw_data_package)
        self.data_cube = raw_data_package["Houston"].astype(np.float32)
        truth = sio.loadmat(r"data/Houston_gt.mat")
        print(truth)
        self.g_truth = truth["Houston_gt"].astype(np.float32)


def load_data():
    data = HongHuRaw().cube
    data_gt =HongHuRaw().truth
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

path_weight = r"/home5/wsq/HyperSIGMA/ImageClassification/seg_weights/"
path_result = r"/home5/wsq/HyperSIGMA/ImageClassification/seg_result/"
data, data_gt = load_data()
height_orgin, width_orgin, bands = data.shape
class_num = np.max(data_gt)
class_num = class_num.astype(int)

data, pca = split_data.apply_PCA(data, num_components=pca_components)

#imshow_IP(data_gt,class_num,height_orgin,width_orgin,r"/home5/wsq/HyperSIGMA/ImageClassification/seg_result/PU_GT.png")

gt_reshape = np.reshape(data_gt, [-1])
train_index, val_index, test_index = split_data.split_data(gt_reshape,
            class_num, train_ratio, val_ratio, train_num, val_num, split_type)

train_index=train_index.astype(int)
val_index=val_index.astype(int)
test_index=test_index.astype(int)
gt_reshape = np.reshape(data_gt, [-1])
class_num = np.max(gt_reshape)
class_num = class_num.astype(int)




train_samples_gt, test_samples_gt, val_samples_gt = create_graph.get_label(gt_reshape,
                                                train_index, val_index, test_index)




train_label_mask, test_label_mask, val_label_mask = create_graph.get_label_mask(train_samples_gt,
                                        test_samples_gt, val_samples_gt, data_gt, class_num)

train_gt = np.reshape(train_samples_gt,[height_orgin,width_orgin])
test_gt = np.reshape(test_samples_gt,[height_orgin,width_orgin])
val_gt = np.reshape(val_samples_gt,[height_orgin,width_orgin])


train_gt_onehot = create_graph.label_to_one_hot(train_gt, class_num)
test_gt_onehot = create_graph.label_to_one_hot(test_gt, class_num)
val_gt_onehot = create_graph.label_to_one_hot(val_gt, class_num)


train_samples_gt=torch.from_numpy(train_samples_gt.astype(np.float32)).to(device)
test_samples_gt=torch.from_numpy(test_samples_gt.astype(np.float32)).to(device)
val_samples_gt=torch.from_numpy(val_samples_gt.astype(np.float32)).to(device)
train_gt_onehot = torch.from_numpy(train_gt_onehot.astype(np.float32)).to(device)
test_gt_onehot = torch.from_numpy(test_gt_onehot.astype(np.float32)).to(device)
val_gt_onehot = torch.from_numpy(val_gt_onehot.astype(np.float32)).to(device)

train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(device)
test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(device)
val_label_mask = torch.from_numpy(val_label_mask.astype(np.float32)).to(device)


img_train, num_H, num_W,data_gt,data = utils.Get_train_and_test_data(img_size, data,data_gt)
height, width, bands = data.shape
img_train = torch.from_numpy(img_train.transpose(0,3,1,2)).type(torch.FloatTensor)
data_train = Data.TensorDataset(img_train)
train_loader = Data.DataLoader(data_train, batch_size=num_H,shuffle=False)
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
spat_net = torch.load((r"/home5/wsq/HyperSIGMA/spat-base.pth"), map_location=torch.device('cpu'))
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
per_net = torch.load((r"/home5/wsq/HyperSIGMA/spec-base.pth"), map_location=torch.device('cpu'))
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




#22201,16
    y_orgin= y_orgin.cpu().numpy()  # 先将 GPU 张量移到 CPU
    y_orgin = np.argmax(y_orgin, axis=1)  # 使用 numpy 进行操作




    # 假设 y_orgin 和 test_samples_gt 是 GPU 上的 Tensor
    # 将它们从 GPU 移动到 CPU，然后转换为 NumPy 数组
    test_gt_onehot = test_gt_onehot.cpu().numpy()
    test_gt_onehot=np.argmax(test_gt_onehot, axis=1)


    test_samples_gt=test_samples_gt.cpu().numpy()




    #y_orgin = y_orgin.cpu().numpy()

    # 类别名称
    class_names = {
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9",
        10: "10",
        11: "11",
        12: "12",
        13: "13",
        14: "14",
        15: "15",
        16: "16",
        17: "17",
        18: "18",
        19: "19",
        20: "20",
        21: "21",
        22: "22"

    }

    # 计算混淆矩阵并手动指定标签
    labels = list(class_names.keys())  # 获取标签 [1, 2, ..., 16]
    cm = confusion_matrix(test_samples_gt, y_orgin, labels=labels)


    #cm = confusion_matrix(test_samples_gt,y_orgin)

    # 类别名称
    #class_names = ['Alfalfa', 'Corn no till', 'Corn till','pasture ','trees ','Hay windrowed','Oats ','Soybean no till ','Soybean till','Wheat','Woods','Buildings-Grass','Stone-Steel-Towers','Residential','Commercial','Road']

    # 打印混淆矩阵
    #print("Confusion Matrix:")
    #print(cm)

    # 绘制混淆矩阵的热力图
    plt.figure(figsize=(10, 10))
    #sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
     #           xticklabels=class_names, yticklabels=class_names)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[class_names[i] for i in range(1, 23)],
                yticklabels=[class_names[i] for i in range(1, 23)])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix Heatmap')
    plt.show()






    # 假设 y 的形状为 (65536, 1)
    y_orgin = y_orgin.reshape(height_orgin,width_orgin, -1)  # 重塑为 (149, 149, 1)
    test_gt_onehot = test_gt_onehot.reshape(height_orgin,width_orgin, -1)

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

imshow_IP(y_orgin,class_num,height_orgin,width_orgin,r"/home5/wsq/HyperSIGMA/ImageClassification/seg_result/PU_classification_result.png")


imshow_IP(data_gt,class_num,height_orgin,width_orgin,r"/home5/wsq/HyperSIGMA/ImageClassification/seg_result/PU_GT.png")


#test-gt(149,149)


# 可视化地物分割结果
#fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#segmentation_map = np.zeros((height, width))

#for (r, c), label in zip(np.argwhere(test_gt > 0), torch.argmax(y, dim=1).cpu().numpy()):
#    segmentation_map[r, c] = label

#cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#800000", "#808000", "#800080", "#008080", "#FFC0CB", "#FFD700", "#ADFF2F", "#00BFFF", "#FF4500", "#8A2BE2"])
#ax.imshow(segmentation_map, cmap=cmap)
#ax.set_title("Segmentation Map")
#ax.axis("off")
#plt.show()