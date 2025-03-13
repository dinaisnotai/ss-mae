import torch
import random
import numpy as np
import torch.nn as nn
from scipy.io import loadmat
from sklearn.decomposition import PCA
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

import h5py
from sklearn.decomposition import IncrementalPCA

# def min_max(x):
#     x = x.astype(np.float32)
#     min = np.min(x)
#     max = np.max(x)
#     return (x - min) / (max - min)

def min_max(x):
    result = np.empty_like(x, dtype=np.float32)
    for i in range(x.shape[0]):  # 逐波段处理
        band = x[i]
        min_val = np.min(band)
        max_val = np.max(band)
        result[i] = (band - min_val) / (max_val - min_val)
    return result


# 设置生成随机数
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# 把三维data降维为二维
def applyPCA(data, n_components):
    h, w, b = data.shape
    data = data.astype(np.float32)
    pca = PCA(n_components=n_components)
    data = np.reshape(pca.fit_transform(np.reshape(data, (-1, b))), (h, w, -1))
    return data



class HXDataset(Dataset):

    # 初始化HXDataset对象
    def __init__(self, hsi, hsi_pca, X, pos, windowSize, gt=None, transform=None, train=False):
        modes = ['symmetric', 'reflect']
        self.train = train
        self.pad = windowSize // 2
        self.windowSize = windowSize
        # hsi = hsi.astype(np.float32)
        # 高光谱、PCA降维后图像，和SAR都被pad处理边界，保证窗口不会超出数组边界
        self.hsi = np.pad(hsi, ((self.pad, self.pad),
                                (self.pad, self.pad), (0, 0)), mode=modes[windowSize % 2])
        self.hsi_pca = np.pad(hsi_pca, ((self.pad, self.pad),
                                        (self.pad, self.pad), (0, 0)), mode=modes[windowSize % 2])
        self.X = None
        if (len(X.shape) == 2):
            self.X = np.pad(X, ((self.pad, self.pad),
                                (self.pad, self.pad)), mode=modes[windowSize % 2])
        elif (len(X.shape) == 3):
            self.X = np.pad(X, ((self.pad, self.pad),
                                (self.pad, self.pad), (0, 0)), mode=modes[windowSize % 2])

        self.pos = pos
        self.gt = None
        if gt is not None:
            self.gt = gt
        if transform:
            self.transform = transform

    def __getitem__(self, index):
        # 根据给定的索引和窗口大小，将HSI、hsi_pca和SAR图像在同一位置的数据进行组合
        # selt.pos[index,;]提供坐标，确保窗口来自同一地理位置（相对坐标）
        h, w = self.pos[index, :]
        hsi = self.hsi[h: h + self.windowSize, w: w + self.windowSize]
        hsi_pca = self.hsi_pca[h: h + self.windowSize, w: w + self.windowSize]
        X = self.X[h: h + self.windowSize, w: w + self.windowSize]
        if self.transform:
            # 如果true的话，就对三个同时进行变换，用于数据增强
            hsi = self.transform(hsi).float()
            hsi_pca = self.transform(hsi_pca).float()
            X = self.transform(X).float()
            trans = [transforms.RandomHorizontalFlip(1.),
                     transforms.RandomVerticalFlip(1.)]
            if self.train:
                if random.random() < 0.5:
                    # 以50%的概率随机反转
                    i = random.randint(0, 1)
                    hsi = trans[i](hsi)
                    X = trans[i](X)
                    hsi_pca = trans[i](hsi_pca)
        if self.gt is not None:
            gt = torch.tensor(self.gt[h, w] - 1).long()
            return hsi.unsqueeze(0), X, gt, hsi_pca.unsqueeze(0)
        return hsi.unsqueeze(0), X, h, w, hsi_pca.unsqueeze(0)

    def __len__(self):
        return self.pos.shape[0]




def load_hsi_v73(filename):
    """Load a specific dataset from a MATLAB v7.3 file using h5py."""
    # key='hyperspectral_image_filtered'
    # with h5py.File(filename, 'r') as f:
    #     if key in f:
    #         return np.array(f[key])
    #     else:
    #         raise KeyError(f"Key '{key}' not found in the MATLAB file.")
    with h5py.File(filename, 'r') as h5_file:
        hyperspectral_matrix = h5_file['hyperspectral_matrix'][:]  # 读取数据集
        return np.array(hyperspectral_matrix)

def load_lidar(filename):

    with h5py.File(filename, 'r') as h5_file:
        hyperspectral_matrix = h5_file['lidar_matrix'][:]  # 读取数据集
        return np.array(hyperspectral_matrix)

def load_gt(filename):
    with h5py.File(filename, 'r') as h5_file:
        hyperspectral_matrix = h5_file['ground_truth'][:]  # 读取数据集
        return np.array(hyperspectral_matrix)

class CustomDataset(Dataset):
    def __init__(self, hsi, hsi_pca, lidar, pos, windowSize, labels=None, transform=None, train=False):
        """
        初始化数据集，只保留 hsi 和 lidar 中都有有效像素的点，并且这些点的坐标一致。

        参数:
            hsi (numpy.ndarray 或 torch.Tensor): 高光谱图像数据，形状为 (H, W, C)。
            lidar (numpy.ndarray 或 torch.Tensor): 激光雷达数据，形状为 (H, W) 或 (H, W, 1)。
            labels (numpy.ndarray 或 torch.Tensor, 可选): 标签数据，形状为 (H, W)。
        """
        # 将输入数据转换为 PyTorch 张量（如果它们不是张量）
        hsi = torch.from_numpy(hsi) if isinstance(hsi, np.ndarray) else hsi
        hsi_pca = torch.from_numpy(hsi_pca) if isinstance(hsi_pca, np.ndarray) else hsi_pca
        lidar = torch.from_numpy(lidar) if isinstance(lidar, np.ndarray) else lidar
        if labels is not None:
            labels = torch.from_numpy(labels) if isinstance(labels, np.ndarray) else labels


        # 打印数据形状
        print("HSI shape:", hsi.shape)
        print("LiDAR shape:", lidar.shape)
        if labels is not None:
            print("Labels shape:", labels.shape)
            labels = labels.squeeze(0)  # 去掉第一个维度
            print("Labels shape:", labels.shape)
            if len(labels.shape) == 3 and labels.shape[0] == 1:
                labels = labels.squeeze(0)  # 去掉第一个维度

        if len(labels.shape)==1:
            labels=labels.reshape(-1,1)
        # 确保 hsi 和 lidar 的形状匹配
        assert hsi.shape[:2] == lidar.shape[:2], "hsi 和 lidar 的空间维度（高度和宽度）必须相同"
        assert len(labels.shape) == 2, "labels的维度不为2"
        # 填充数据以处理边界
        self.pad = windowSize // 2
        self.windowSize = windowSize
        modes = ['symmetric', 'reflect']
        self.hsi = np.pad(hsi, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode=modes[windowSize % 2])
        self.hsi_pca = np.pad(hsi_pca, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode=modes[windowSize % 2])
        if len(lidar.shape) == 2:
            self.lidar = np.pad(lidar, ((self.pad, self.pad), (self.pad, self.pad)), mode=modes[windowSize % 2])
        elif len(lidar.shape) == 3:
            self.lidar = np.pad(lidar, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode=modes[windowSize % 2])

        self.pos = pos
        self.labels = labels
        self.transform = transform
        self.train = train

    def __len__(self):
        return self.pos.shape[0]

    def __getitem__(self, index):
        # 根据给定的索引和窗口大小，提取数据
        if len(self.labels.shape) == 3 and self.labels.shape[0] == 1:
            print(self.labels.shape,"labels出错")
            self.labels = self.labels.squeeze(0)  # 去掉第一个维度
        h, w = self.pos[index, :]
        h_padded = h + self.pad  # 添加填充偏移
        w_padded = w + self.pad  # 添加填充偏移
        if h >= self.labels.shape[0] or w >= self.labels.shape[1]:
            raise IndexError(f"Index ({h}, {w}) is out of bounds for labels with shape {self.labels.shape}")
        # print(f"Index: {index}, h: {h}, w: {w}, gt shape: {self.labels.shape}")
        # hsi = self.hsi[h: h + self.windowSize, w: w + self.windowSize]
        # hsi_pca = self.hsi_pca[h: h + self.windowSize, w: w + self.windowSize]
        # lidar = self.lidar[h: h + self.windowSize, w: w + self.windowSize]

        hsi = self.hsi[h_padded: h_padded + self.windowSize, w_padded: w_padded + self.windowSize]
        hsi_pca = self.hsi_pca[h_padded: h_padded + self.windowSize, w_padded: w_padded + self.windowSize]
        lidar = self.lidar[h_padded: h_padded + self.windowSize, w_padded: w_padded + self.windowSize]

        # 数据增强
        if self.transform:
            hsi = self.transform(hsi).float()
            hsi_pca = self.transform(hsi_pca).float()
            lidar = self.transform(lidar).float()
            if self.train:
                trans = [transforms.RandomHorizontalFlip(1.), transforms.RandomVerticalFlip(1.)]
                if random.random() < 0.5:
                    i = random.randint(0, 1)
                    hsi = trans[i](hsi)
                    hsi_pca = trans[i](hsi_pca)
                    lidar = trans[i](lidar)

        # 返回数据
        if self.labels is not None:
            # gt_value = self.labels[h, w]
            gt = torch.tensor(self.labels[h, w] ).long()
            # gt = torch.tensor(gt_value - 1 if gt_value > 0 else 0).long()
            # print(f"Label value: {gt}, min: {self.labels.min()}, max: {self.labels.max()}")
            return hsi.unsqueeze(0), lidar, gt, hsi_pca.unsqueeze(0)
        return hsi.unsqueeze(0), lidar, h,w,hsi_pca.unsqueeze(0)

# def custom_collate_fn(batch):
#     # 将batch中的数据堆叠起来
#     hsi = torch.stack([item[0] for item in batch])
#     lidar = torch.stack([item[1] for item in batch])
#     if len(batch[0]) > 2:
#         labels = torch.stack([item[2] for item in batch])
#         return hsi, lidar, labels
#     return hsi, lidar

def custom_collate_fn(batch):
    hsi = torch.stack([item[0] for item in batch])
    lidar = torch.stack([item[1] for item in batch])
    tr_labels = torch.stack([item[2] for item in batch])
    hsi_pca = torch.stack([item[3] for item in batch])
    return hsi, lidar, tr_labels, hsi_pca





def load_pca(filename):
    with h5py.File(filename, 'r') as h5_file:
        hyperspectral_matrix = h5_file['pca_data'][:]  # 读取数据集
        return np.array(hyperspectral_matrix)


def getData(hsi_path, X_path, gt_path, index_path, keys, channels, windowSize, batch_size, num_workers, args):
    '''
    hsi: Hyperspectral image data
    X: Other modal data
    gt: Ground truth labels, where 0 represents unlabeled
    train_index: Indices for training data
    test_index: Indices for testing data
    pretrain_index: Indices for pretraining data
    trntst_index: Indices for training and testing data, used for visualizing labeled data
    all_index: Indices for all data, including unlabeled data, used for visualizing all data or pretraining

    '''



    if(keys==['tlse_hsi', 'tlse_lidar', 'tlse_gt', 'tlse_train', 'tlse_test', 'tlse_all']):
        hsi=load_hsi_v73(hsi_path)
        X=load_lidar(X_path)
        gt=load_gt(gt_path)
    else:
        hsi = loadmat(hsi_path)[keys[0]]
        # sar
        X = loadmat(X_path)[keys[1]]
        # ground truth
        gt = loadmat(gt_path)[keys[2]]






    # 训练索引
    train_index = loadmat(index_path)[keys[3]]
    # 测试索引
    test_index = loadmat(index_path)[keys[4]]
    # 合并上面两个索引
    trntst_index = np.concatenate((train_index, test_index), axis=0)
    # 如果设置了预训练标志 args.is_pretrain，则对全部索引 all_index 进行随机打乱
    all_index = loadmat(index_path)[keys[5]]

    if args.is_pretrain:
        np.random.shuffle(all_index)
        # 创建预训练索引数组
        pretrain_index = np.zeros((args.pretrain_num, 2), dtype=np.int32)
        count = 0
        for i in all_index:
            # print(i)
            if 15 < i[0] < hsi.shape[0] - 15:
                if 15 < i[1] < hsi.shape[1] - 15:
                    pretrain_index[count] = i
                    count += 1
                    if count == args.pretrain_num:
                        break

    # 归一化

    # 高光谱
    hsi = min_max(hsi)
    hsi = np.transpose(hsi, (1, 2, 0))
    X = min_max(X)
    print(X.shape)
    print(hsi.shape)

    # PCA is used to reduce the dimensionality of the HSI
    hsi_pca = applyPCA(hsi, channels)   #降维
    # hsi_pca=apply_incremental_pca(hsi,channels)

    pretrain_loader = None

    # Build Dataset 构建数据集
    if args.is_pretrain:
        HXpretrainset = HXDataset(hsi, hsi_pca, X, pretrain_index,
                                  windowSize, transform=ToTensor(), train=True)
    HXtrainset = HXDataset(hsi, hsi_pca, X, train_index,
                           windowSize, gt, transform=ToTensor(), train=True)
    HXtestset = HXDataset(hsi, hsi_pca, X, test_index,
                          windowSize, gt, transform=ToTensor())
    HXtrntstset = HXDataset(hsi, hsi_pca, X, trntst_index,
                            windowSize,gt, transform=ToTensor())
    HXallset = HXDataset(hsi, hsi_pca, X, all_index,
                         windowSize,gt, transform=ToTensor())

    # Build Dataloader 创建训练、测试加载器
    if args.is_pretrain:
        pretrain_loader = DataLoader(
            HXpretrainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    if(keys==['tlse_hsi', 'tlse_lidar', 'tlse_gt', 'tlse_train', 'tlse_test', 'tlse_all']):
        train_dataset = CustomDataset(hsi, hsi_pca, X, train_index,
                           windowSize, gt, transform=ToTensor(), train=True)
        test_dataset = CustomDataset(hsi, hsi_pca, X, test_index,
                              windowSize, gt, transform=ToTensor())
        trntst_dataset = CustomDataset(hsi, hsi_pca, X, trntst_index,
                                windowSize, gt, transform=ToTensor())
        all_dataset = CustomDataset(hsi, hsi_pca, X, all_index,
                             windowSize, gt, transform=ToTensor())

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        trntst_loader = DataLoader(
            trntst_dataset, batch_size = 1,shuffle = True,num_workers=num_workers,drop_last=True)
        all_loader = DataLoader(
            all_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
        print(f"all_loader 总样本数: {len(all_loader.dataset)}")
    else:
        train_loader = DataLoader(
            HXtrainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        test_loader = DataLoader(
            HXtestset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
        trntst_loader = DataLoader(
            HXtrntstset, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=True)
        all_loader = DataLoader(
            HXallset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

        #

     # 表示构建数据集成功
    print("Success!")
    return pretrain_loader, train_loader, test_loader, trntst_loader, all_loader


def getHouston2018Data(hsi_path, lidar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers, args):
    print("Houston2018!")
    # Houston2018 mat keys
    houston2018_keys = ['houston_hsi', 'houston_lidar', 'houston_gt', 'houston_train', 'houston_test', 'houston_all']

    return getData(hsi_path, lidar_path, gt_path, index_path, houston2018_keys, channels, windowSize, batch_size,
                   num_workers, args)


def getBerlinData(hsi_path, sar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers, args):
    print("Berlin!")
    # Berlin mat keys
    berlin_keys = ['berlin_hsi', 'berlin_sar', 'berlin_gt', 'berlin_train', 'berlin_test', 'berlin_all']

    return getData(hsi_path, sar_path, gt_path, index_path, berlin_keys, channels, windowSize, batch_size, num_workers,
                   args)


def getAugsburgData(hsi_path, sar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers, args):
    print("Augsburg!")
    # Augsburg mat keys
    augsburg_keys = ['augsburg_hsi', 'augsburg_sar', 'augsburg_gt', 'augsburg_train', 'augsburg_test', 'augsburg_all']

    return getData(hsi_path, sar_path, gt_path, index_path, augsburg_keys, channels, windowSize, batch_size,
                   num_workers, args)

def getTlseData(hsi_path, lidar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers, args):
    print("Tlse!")
    # Houston2018 mat keys
    Tlse_keys = ['tlse_hsi', 'tlse_lidar', 'tlse_gt', 'tlse_train', 'tlse_test', 'tlse_all']

    return getData(hsi_path, lidar_path, gt_path, index_path, Tlse_keys, channels, windowSize, batch_size,
                   num_workers, args)


def getHSData(datasetType, channels, windowSize, batch_size=64, num_workers=0, args=None):
    if (datasetType == "Berlin"):
        hsi_path = "data/Berlin/berlin_hsi.mat"
        sar_path = "data/Berlin/berlin_sar.mat"
        gt_path = "data/Berlin/berlin_gt.mat"
        index_path = "data/Berlin/berlin_index.mat"
        pretrain_loader, train_loader, test_loader, trntst_loader, all_loader = getBerlinData(hsi_path, sar_path, gt_path, index_path,
                                                                             channels, windowSize, batch_size,
                                                                             num_workers, args)
    elif (datasetType == "Augsburg"):
        hsi_path = "data/Augsburg/augsburg_hsi.mat"
        sar_path = "data/Augsburg/augsburg_sar.mat"
        gt_path = "data/Augsburg/augsburg_gt.mat"
        index_path = "data/Augsburg/augsburg_index.mat"
        pretrain_loader, train_loader, test_loader, trntst_loader, all_loader = getAugsburgData(hsi_path, sar_path, gt_path, index_path,
                                                                               channels,
                                                                               windowSize, batch_size, num_workers,
                                                                               args)
    elif (datasetType == "Houston2018"):
        hsi_path = "data/Houston2018/houston_hsi.mat"
        lidar_path = "data/Houston2018/houston_lidar.mat"
        gt_path = "data/Houston2018/houston_gt.mat"
        index_path = "data/Houston2018/houston_index.mat"
        pretrain_loader, train_loader, test_loader, trntst_loader, all_loader = getHouston2018Data(hsi_path, lidar_path, gt_path,
                                                                                  index_path,
                                                                                  channels, windowSize, batch_size,
                                                                                  num_workers, args)

    elif (datasetType == "Tlse"):
        hsi_path = "data/tlse/processed_hsi.h5"
        lidar_path = "data/tlse/processed_lidar.h5"
        gt_path = "data/tlse/processed_gt.h5"
        index_path = "data/tlse/tlse_index.mat"
        pretrain_loader, train_loader, test_loader, trntst_loader, all_loader = getTlseData(hsi_path, lidar_path,
                                                                                                   gt_path,
                                                                                                   index_path,
                                                                                                   channels, windowSize,
                                                                                                   batch_size,
                                                                                                   num_workers, args)

    return pretrain_loader, train_loader, test_loader, trntst_loader, all_loader

def getTlseSmoteData(hsi_path, X_path, gt_path, index_path, channels, windowSize, batch_size, num_workers, args):
    '''
    hsi: Hyperspectral image data
    X: Other modal data
    gt: Ground truth labels, where 0 represents unlabeled
    train_index: Indices for training data
    test_index: Indices for testing data
    pretrain_index: Indices for pretraining data
    trntst_index: Indices for training and testing data, used for visualizing labeled data
    all_index: Indices for all data, including unlabeled data, used for visualizing all data or pretraining
    '''

    # if keys == ['tlse_hsi', 'tlse_lidar', 'tlse_gt', 'tlse_train', 'tlse_test', 'tlse_all']:
    #
    # else:
    #     hsi = loadmat(hsi_path)[keys[0]]
    #     X = loadmat(X_path)[keys[1]]
    #     gt = loadmat(gt_path)[keys[2]]

    hsi = load_hsi_v73(hsi_path)
    X = load_lidar(X_path)
    gt = load_gt(gt_path)

    Tlse_keys = ['tlse_hsi', 'tlse_lidar', 'tlse_gt', 'tlse_train', 'tlse_test', 'tlse_all']
    train_index = loadmat(index_path)[Tlse_keys[3]]
    test_index = loadmat(index_path)[Tlse_keys[4]]
    trntst_index = np.concatenate((train_index, test_index), axis=0)
    all_index = loadmat(index_path)[Tlse_keys[5]]

    if args.is_pretrain:
        np.random.shuffle(all_index)
        pretrain_index = np.zeros((args.pretrain_num, 2), dtype=np.int32)
        count = 0
        for i in all_index:
            if 15 < i[0] < hsi.shape[0] - 15:
                if 15 < i[1] < hsi.shape[1] - 15:
                    pretrain_index[count] = i
                    count += 1
                    if count == args.pretrain_num:
                        break

    hsi = min_max(hsi)
    hsi = np.transpose(hsi, (1, 2, 0))
    lidar = min_max(X)

    # PCA降维
    hsi_pca = applyPCA(hsi, channels)

    # 提取训练数据的特征和标签
    hsi_features = []
    lidar_features = []
    labels = []
    gt=gt.squeeze(0)

    for idx in train_index:
        h, w = idx
        hsi_feature = hsi[h:h + windowSize, w:w + windowSize].flatten()
        lidar_feature = lidar[h:h + windowSize, w:w + windowSize].flatten()
        # print(gt.shape)
        label = gt[h, w]
        hsi_features.append(hsi_feature)
        lidar_features.append(lidar_feature)
        labels.append(label)

    hsi_features = np.array(hsi_features)
    lidar_features = np.array(lidar_features)
    labels = np.array(labels).reshape(-1, 1)
    print(labels.shape)

    # 合并HSI和LiDAR特征
    features = np.concatenate([hsi_features, lidar_features], axis=1)

    # 应用SMOTE
    smote = SMOTE(random_state=42)
    features_resampled, labels_resampled = smote.fit_resample(features, labels)

    # 将SMOTE后的数据重新分割为HSI和LiDAR
    hsi_resampled = features_resampled[:, :hsi_features.shape[1]].reshape(-1, windowSize, windowSize, hsi.shape[2])
    lidar_resampled = features_resampled[:, hsi_features.shape[1]:].reshape(-1, windowSize, windowSize)

    # 创建新的训练索引
    new_train_index = np.array([[i, j] for i in range(hsi_resampled.shape[0]) for j in range(hsi_resampled.shape[1])])

    # 创建新的训练数据集
    train_dataset = CustomDataset(hsi_resampled, hsi_pca, lidar_resampled, new_train_index, windowSize,
                                  labels_resampled, transform=ToTensor(), train=True)
    test_dataset = CustomDataset(hsi, hsi_pca, lidar, test_index, windowSize, gt, transform=ToTensor())
    trntst_dataset = CustomDataset(hsi, hsi_pca, lidar, trntst_index, windowSize, gt, transform=ToTensor())
    all_dataset = CustomDataset(hsi, hsi_pca, lidar, all_index, windowSize, gt, transform=ToTensor())

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             drop_last=True)
    trntst_loader = DataLoader(trntst_dataset, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=True)
    all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    print("Success! SMOTE applied to Tlse dataset.")
    pretrain_loader = None

    return pretrain_loader,train_loader, test_loader, trntst_loader, all_loader