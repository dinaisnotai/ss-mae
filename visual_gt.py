import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import argparse

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from net.VIT.mae import VisionTransfromers as MAEFinetune
from get_dat import get_dataset
from scipy.io import loadmat
import h5py

from utility import accuracy


def visualize_predictions(args, model, all_loader, gt_path):
    """
    仅对有效标签坐标进行预测并可视化对比结果
    """
    model.eval()  # 将模型设置为评估模式
    predictions = []  # 存储预测结果
    if torch.cuda.is_available():
        device = torch.device("cuda")  # 如果可用，使用CUDA
    else:
        device = torch.device("cpu")

    model.eval()
    predictions = []
    true_labels = []

    # 加载真实标签矩阵
    with h5py.File(gt_path, 'r') as f:
        gt = f['ground_truth'][:].astype(np.int32)
    gt = gt.squeeze()  # 去除冗余维度

    # 获取有效标签的坐标（非背景）
    valid_coords = np.argwhere(gt > 0)
    print(f"有效标签数量: {len(valid_coords)}")
    print(f"all_loader 总样本数: {len(all_loader.dataset)}")

    # 仅对有效坐标进行预测
    recorded_positions = []
    with torch.no_grad():
        for batch_idx, (hsi, lidar, labels, hsi_pca) in enumerate(all_loader):
            hsi = hsi.to(device)
            hsi = hsi[:, 0, :, :, :]
            hsi_pca = hsi_pca.to(device)
            lidar = lidar.to(device)

            outputs, _ = model(hsi, lidar, hsi_pca)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            labels = labels[:len(preds)]  # 关键修复
            predictions.extend(preds)
            true_labels.extend(labels.numpy())
            # print("preds shape:", preds.shape)
            # print("true_labels shape:", labels.shape)
            print(f"当前 predictions 数量: {len(predictions)}")
            print(np.unique(labels), np.unique(preds))


            print("pred", np.unique(preds))
            print("accuracy", accuracy_score(preds, labels))
            # recorded_positions.extend(self.pos[batch_idx * batch_size: (batch_idx + 1) * batch_size])


    # 转换为数组
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    print(accuracy_score(true_labels, predictions))

    # 计算分类报告（排除背景）
    classification = classification_report(true_labels, predictions, digits=4, zero_division=0)
    print("分类报告（仅有效标签）:\n", classification)


    # 生成预测和真实标签图像（仅有效区域）
    pred_image = np.zeros_like(gt, dtype=np.uint8)
    true_image = np.zeros_like(gt, dtype=np.uint8)

    all_index = loadmat("data/tlse/tlse_index.mat")['tlse_all']

    valid_count = 0
    for h, w in all_index:
        if h < gt.shape[0] and w < gt.shape[1] and gt[h, w] > 0:
            valid_count += 1
    print(f"all_index 中有效坐标数量: {valid_count}")

    idx = 0  # 初始化 idx
    # print(f"gt 的最小值: {np.min(gt)}")
    # print(f"gt 的最大值: {np.max(gt)}")
    for h, w in all_index:
        if h < gt.shape[0] and w < gt.shape[1]:
            if gt[h, w] > 0:  # 仅填充有效标签位置
                pred_image[h, w] = predictions[idx]   # 预测类别+1对齐
                true_image[h, w] = gt[h, w]  # 真实标签
                 # 每次循环后累加 idx
                # if gt[h,w]!=true_labels[idx]:
                #     print(gt[h,w],true_labels[idx])

                idx += 1
                # if(idx >= len(predictions)):
                #     break


    # 可视化对比
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(true_image, cmap='jet', vmin=0, vmax=args.num_classes)
    plt.title("Ground Truth")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(pred_image, cmap='jet', vmin=0, vmax=args.num_classes)
    plt.title("Predicted Labels")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    error_map = np.where(pred_image != true_image, 1, 0)  # 错误区域标记为1
    plt.imshow(error_map, cmap='gray')
    plt.title("correct Map ")

    plt.tight_layout()
    plt.show()


# 在主函数中调用修改后的可视化函数
if __name__ == '__main__':
    # ...（原有参数设置代码不变）

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
        batch_size = 256


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
    save_dir = os.path.join('model', 'train', '20250125_231437')
    model_path = os.path.join(save_dir, 'best_model.pth')
    checkpoint = torch.load(model_path, map_location="cpu")
    model = model.to(device)
    model.cuda(device=device)

    model.load_state_dict(checkpoint['state_dict'])


    # 调用时传入真实标签路径
    gt_path = "data/tlse/processed_gt.h5"
    pretrain_loader, train_loader, test_loader, trntst_loader, all_loader = get_dataset(args)

    # 调用可视化函数
    visualize_predictions(args, model, all_loader, gt_path)