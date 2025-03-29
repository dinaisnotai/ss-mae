import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import argparse

from sklearn.metrics import accuracy_score

from net.VIT.mae import VisionTransfromers as MAEFinetune
from get_dat import get_dataset
from scipy.io import loadmat
import h5py


def visualize_predictions(args, model, all_loader, gt_path):

        """
        对所有点进行预测并可视化结果
        """
        model.eval()  # 将模型设置为评估模式
        predictions = []  # 存储预测结果
        if torch.cuda.is_available():
            device = torch.device("cuda")  # 如果可用，使用CUDA
        else:
            device = torch.device("cpu")

        # visual.py 中的 visualize_predictions 函数
        for batch_idx, (hsi, lidar, labels, hsi_pca) in enumerate(all_loader):
            print(f"Batch {batch_idx} - HSI shape: {hsi.shape}, LiDAR shape: {lidar.shape}")
            break  # 仅检查第一个批次


        with torch.no_grad():
            for batch_idx, (hsi, lidar, labels, hsi_pca) in enumerate(all_loader):
                hsi = hsi.to(device)
                hsi = hsi[:, 0, :, :, :]  # 调整维度
                hsi_pca = hsi_pca.to(device)
                lidar = lidar.to(device)

                # 进行预测
                outputs, _ = model(hsi, lidar, hsi_pca)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()  # 获取预测类别
                predictions.extend(preds)

        # 将预测结果转换为图像格式
        predictions = np.array(predictions)

        # 获取原始数据的形状（假设数据是二维的）
        hsi_path = "data/tlse/processed_hsi.h5"  # 替换为你的数据路径
        with h5py.File(hsi_path, 'r') as h5_file:
            hsi_data = h5_file['hyperspectral_matrix'][:]
        bands, height, width = hsi_data.shape

        # 创建一个全零的预测图像
        pred_image = np.zeros((height, width), dtype=np.uint8)

        # 获取整个图像的坐标

        # all_indices = np.array([[h, w] for h in range(height) for w in range(width)])

        all_index = loadmat("data/tlse/tlse_index.mat")['tlse_all']

        # 修改循环方式，直接使用索引顺序填充

        for idx, (h, w) in enumerate(all_index):
            pred_image[h, w] = predictions[idx] + 1  # 假设需要+1对齐标签 # 预测类别+1对齐


        # 可视化预测结果
        plt.figure(figsize=(12, 6))
        plt.imshow(pred_image, cmap='jet')
        plt.title("Predicted Labels")
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        plt.savefig('pic.png')


# 在主函数中调用可视化函数
if __name__ == '__main__':
    # 加载最佳模型
    class Args:
        is_train = 0
        is_load_pretrain = 0
        is_pretrain = 1
        is_test = 0
        model_file = 'model'
        size_SA = 49
        channel_number = 291
        epoch = 500
        pca_num = 30
        mask_ratio = 0.7
        crop_size = 7
        device = "cuda:0"
        dataset = 'Tlse'
        num_classes = 12
        pretrain_num = 400000
        patch_size = 1
        finetune = 0
        mae_pretrain = 1
        depth = 3
        head = 16
        dim = 256
        model_name = None
        warmup_epochs = 5
        test_interval = 5
        optimizer_name = "adamw"
        lr = 1e-4
        cosine = 0
        weight_decay = 5e-2
        batch_size = 128


    args = Args()

    model = MAEFinetune(
        channel_number=args.channel_number,
        img_size=args.crop_size,
        patch_size=args.patch_size,
        embed_dim=args.dim,
        depth=args.depth,
        num_heads=args.head,
        num_classes=args.num_classes,
        args=args
    )

    device = 'cuda'
    save_dir = os.path.join('model', 'train', '20250123_231311')
    model_path = os.path.join(save_dir, 'best_model.pth')
    checkpoint = torch.load(model_path, map_location="cpu")
    model = model.to(device)
    model.cuda(device=device)

    model.load_state_dict(checkpoint['state_dict'])

    pretrain_loader, train_loader, test_loader, trntst_loader, all_loader = get_dataset(args)

    # 调用可视化函数
    visualize_predictions(args, model, test_loader)