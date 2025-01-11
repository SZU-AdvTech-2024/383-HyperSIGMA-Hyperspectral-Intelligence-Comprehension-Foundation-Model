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

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
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
    data_gt =IndianRaw().truth
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


# 移动到 CPU 并展平
y = y.cpu().numpy()
y = np.argmax(pred, axis=1)  # 形状为 (22201,)
# 将预测结果和真实标签重塑为二维图像
predicted_map  = y.reshape(height, width)
#ground_truth_map = test_gt.reshape(height, width)
#ground_truth_map=ground_truth_map.cpu().numpy()


def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1]*2.0/dpi, ground_truth.shape[0]*2.0/dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    #不显示坐标轴
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)
    return 0


def generate_colormap(class_num):
    np.random.seed(42)
    colors = np.random.rand(class_num, 3)
    return ListedColormap(colors)


colormap = generate_colormap(class_num)


# Step 4: 绘制分类图
def plot_classification_map(predicted_map, ground_truth_map, colormap, save_path=None):
    plt.figure(figsize=(12, 6))

    # 绘制预测分类结果
    plt.subplot(1, 2, 1)
    plt.title("Predicted Classification")
    plt.imshow(predicted_map, cmap=colormap)
    plt.colorbar()
    plt.axis("off")

    # 绘制地面真实标签
    plt.subplot(1, 2, 2)
    plt.title("Ground Truth")
    plt.imshow(ground_truth_map, cmap=colormap)
    plt.colorbar()
    plt.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

# 调用绘图函数
plot_classification_map(predicted_map, data_gt, colormap, save_path="/home5/wsq/HyperSIGMA/ImageClassification/seg_result/classification_result.png")