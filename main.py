from utility import accuracy
import math
import time
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from utils.optimizer_step import Optimizer
import datetime
from utility import output_metric
from collections import defaultdict
import argparse
from collections import Counter
from visual_gt import visualize_predictions
parser = argparse.ArgumentParser("HSI")

# ---- stage
parser.add_argument('--is_train', default=0, type=int)
parser.add_argument('--is_load_pretrain', default=0, type=int)
parser.add_argument('--is_pretrain', default=1, type=int)
parser.add_argument('--is_test', default=1, type=int)
parser.add_argument('--model_file', default='model', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# ---- network parameter
parser.add_argument('--size_SA', default=49, type=int, help='the size of spatial attention')
parser.add_argument('--channel_num', default=51, type=int, help="the size of spectral attention (berlin 248, augsburg "
                                                                "188, houston 2018 51)")
parser.add_argument('--epoch', default=500, type=int)
parser.add_argument('--pca_num', default=20, type=int)
parser.add_argument('--mask_ratio', default=0.3, type=float)
parser.add_argument('--crop_size', type=int, default=7)

# ----- data
parser.add_argument('--device', default="cuda:0", type=str)
parser.add_argument('--dataset', default='Houston2018', type=str,
                    help='Houston2018 Berlin Augsburg')
parser.add_argument('--num_classes', default=20, type=int, help="berlin 8, augsburg 7, houston 2018 20")
parser.add_argument('--pretrain_num', default=50000, type=int)

# --- vit
parser.add_argument('--patch_size', default=1, type=int)
parser.add_argument('--finetune', default=0, type=int)
parser.add_argument('--mae_pretrain', default=1, type=int)
parser.add_argument('--depth', default=2, type=int)
parser.add_argument('--head', default=4, type=int)
parser.add_argument('--dim', default=256, type=int)

# ---- train
parser.add_argument('--model_name', type=str)
parser.add_argument('--warmup_epochs', default=5, type=int)
parser.add_argument('--test_interval', default=2, type=int)
parser.add_argument('--optimizer_name', default="adamw", type=str)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--cosine', default=0, type=int)
parser.add_argument('--weight_decay', default=5e-2, type=float)
parser.add_argument('--batch_size', default=64, type=int)

args = parser.parse_args()

from get_dat import get_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

device = torch.device(args.device if torch.cuda.is_available() else "cpu")


from net.VIT.mae import MAEVisionTransformers as MAE
from net.VIT.mae import VisionTransfromers as MAEFinetune
from loss.mae_loss import MSELoss, build_mask_spa, build_mask_chan


pretrain_loader, train_loader, test_loader, trntst_loader, all_loader = get_dataset(args)


def min_max(input):
    return (input - input.min()) / (input.max() - input.min()) * 255


def Pretrain(args,
             scaler,
             model,
             criterion,
             optimizer,
             epoch,
             batch_iter,
             batch_size
             ):
    """Traing with the batch iter for get the metric
    """

    total_loss = 0
    n = 0
    loader_length = len(pretrain_loader)
    print("pretrain_loader-------------", loader_length)
    for batch_idx, (hsi, lidar, _, _, hsi_pca) in enumerate(pretrain_loader):
        n = n + 1
        # TODO: add the layer-wise lr decay
        if args.cosine:
            # cosine learning rate
            lr = cosine_learning_rate(
                args, epoch, batch_iter, optimizer, batch_size
            )
        else:
            # step learning rate
            lr = step_learning_rate(
                args, epoch, batch_iter, optimizer, batch_size
            )

        # forward
        hsi = hsi.to(device)
        hsi = hsi[:, 0, :, :, :]
        lidar = lidar.to(device)

        # 此时已经是双输入（光谱和空间双输入）
        # 调用mae模型的forward方法
        outputs_spa, mask_index_spa, outputs_chan, mask_index_chan = model(torch.cat((hsi, lidar), 1))  #torch.cat拼接两个张良
        mask_spa = build_mask_spa(mask_index_spa, args.patch_size, args.crop_size)
        mask_chan = build_mask_chan(mask_index_chan, channel_num=args.channel_num, patch_size=args.patch_size)
        losses = criterion(outputs_spa, torch.cat((hsi, lidar), 1), mask_spa) + criterion(outputs_chan,
                                                                                          torch.cat((hsi, lidar), 1),
                                                                                          mask_chan.unsqueeze(-1))

        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss = total_loss + losses.data.item()
        batch_iter += 1

        print(
            "Epoch:", epoch,
            " batch_idx:", batch_idx,
            " batch_iter:", batch_iter,
            " losses:", losses.data.item(),
            " lr:", lr
        )
    print(
        "Epoch:", epoch,
        " losses:", total_loss / n,
        " lr:", lr
    )
    return total_loss / n, batch_iter, scaler

class_weights = {
    0: 0.0190,
    1: 0.3027,
    2: 0.0017,
    3: 0.0322,
    4: 0.0237,
    5: 0.0046,
    6: 0.0430,
    7: 0.8151,
    8: 2.0531,
    9: 2.090,
    10: 2.6914,
    11: 0.0104
}

def Train(args,
          scaler,
          model,
          criterion,
          optimizer,
          epoch,
          batch_iter,
          batch_size
          ):
    """Traing with the batch iter for get the metric
    """

    acc = 0
    n = 0
    class_counts = defaultdict(int)

    def custom_loss(output, target):
        # 忽略类别为0的样本
        mask = target != 0
        loss = nn.CrossEntropyLoss()(output[mask], target[mask])
        return loss


    # 假设有效像素坐标存储在数据集中

    # 计算 unique_classes
    labels = train_loader.dataset.labels
    unique_classes = np.unique(labels[labels > 0])

    for batch_idx, batch_data in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        for i, data in enumerate(batch_data):
            print(f"  Data {i}: Shape={data.shape}, Type={type(data)}")
        break  # 只打印第一个批次

    for batch_idx, (hsi, lidar, tr_labels, hsi_pca) in enumerate(train_loader):
        labels = tr_labels.numpy()
        for label in labels:
            class_counts[label] += 1
        n = n + 1
        # hsi, lidar = preprocess_batch(hsi, lidar)
        hsi = hsi.to(device)
        hsi = hsi[:, 0, :, :, :]
        hsi_pca = hsi_pca.to(device)
        lidar = lidar.to(device)
        tr_labels = tr_labels.to(device)
        # TODO: add the layer-wise lr decay
        if args.cosine:
            # cosine learning rate
            lr = cosine_learning_rate(
                args, epoch, batch_iter, optimizer, batch_size
            )
        else:
            # step learning rate
            lr = step_learning_rate(
                args, epoch, batch_iter, optimizer, batch_size
            )
            # forward

        outputs, _ = model(hsi, lidar, hsi_pca)


        # 在训练过程中使用自定义损失函数
        losses = custom_loss(outputs, tr_labels)
        # losses = criterion(outputs, tr_labels)

        optimizer.zero_grad()

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_acc, _ = accuracy(outputs, tr_labels)

        acc = acc + batch_acc[0]

        batch_iter += 1

        print(
            "Epoch:", epoch,
            " batch_idx:", batch_idx,
            " batch_iter:", batch_iter,
            " batch_acc:", batch_acc[0],
            "loss:",losses.data.item(),
            " lr:", lr
        )

    print("训练集类别分布:")
    for cls in sorted(class_counts.keys()):
        print(f"类别 {cls}: {class_counts[cls]} 个样本")

    # 检查是否存在样本数为 0 的类别

    missing_classes = [cls for cls in unique_classes if class_counts.get(cls, 0) == 0]
    if missing_classes:
        print(f"警告：以下类别在训练集中没有样本: {missing_classes}")
    else:
        print("训练集包含所有类别！")

    print(
        "Epoch:", epoch,
        " acc:", acc / n,
        " lr:", lr
    )
    return acc / n, batch_iter, scaler


def val(
        args,
        model
):
    """Validation and get the metric
    """
    batch_acc_list = []
    count = 0
    with torch.no_grad():
        for batch_idx, (hsi, lidar, tr_labels, hsi_pca) in enumerate(test_loader):

            hsi = hsi.to(device)
            hsi = hsi[:, 0, :, :, :]
            lidar = lidar.to(device)
            hsi_pca = hsi_pca.to(device)
            tr_labels = tr_labels.to(device)

            if args.is_pretrain == True:
                outputs,_=model(hsi)
            else:
                outputs, _ = model(hsi, lidar, hsi_pca)

            batch_accuracy, _ = accuracy(outputs, tr_labels)

            batch_acc_list.append(batch_accuracy[0])

            if count == 0:
                y_pred_test = np.argmax(outputs.detach().cpu().numpy(), axis=1)
                gty = tr_labels.detach().cpu().numpy()
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, np.argmax(outputs.detach().cpu().numpy(), axis=1)))  #
                gty = np.concatenate((gty, tr_labels.detach().cpu().numpy()))

    OA2, AA_mean2, Kappa2, AA2 = output_metric(gty, y_pred_test)
    print("val_pred_unique",np.unique(y_pred_test))
    print("val_label_unique", np.unique(gty))
    classification = classification_report(gty, y_pred_test, digits=4)
    print(classification)
    print("OA2=", OA2)
    print("AA_mean2=", AA_mean2)
    print("Kappa2=", Kappa2)
    print("AA2=", AA2)
    epoch_acc = np.mean(batch_acc_list)

    print("Epoch_mean_accuracy:" % epoch_acc)

    cm = confusion_matrix(gty, y_pred_test)
    print("Confusion Matrix:")
    print(cm)


    return epoch_acc


def step_learning_rate(args, epoch, batch_iter, optimizer, train_batch):
    total_epochs = args.epoch
    warm_epochs = args.warmup_epochs
    if epoch <= warm_epochs:
        lr_adj = (batch_iter + 1) / (warm_epochs * train_batch)
        # lr_adj = 1.
    elif epoch < int(0.6 * total_epochs):
        lr_adj = 1.
    elif epoch < int(0.8 * total_epochs):
        lr_adj = 1e-1
    elif epoch < int(1 * total_epochs):
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * lr_adj
    return args.lr * lr_adj


def cosine_learning_rate(args, epoch, batch_iter, optimizer, train_batch):
    """Cosine Learning rate
    """
    total_epochs = args.max_epochs
    warm_epochs = args.warmup_epochs
    if epoch <= warm_epochs:
        lr_adj = (batch_iter + 1) / (warm_epochs * train_batch) + 1e-6
    else:
        lr_adj = 1 / 2 * (1 + math.cos(batch_iter * math.pi /
                                       ((total_epochs - warm_epochs) * train_batch)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * lr_adj
    return args.lr * lr_adj


def translate_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'module' in key:
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        crr = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            acc = correct_k.mul_(1 / batch_size).item()
            res.append(acc)  # unit: percentage (%)
            crr.append(correct_k)
        return res, crr


def preprocess_data(hsi_data, lidar_data):
    """
    预处理HSI和LiDAR数据,只保留两者都有效的像素点
    """
    # 找出HSI和LiDAR中非零的像素位置
    valid_mask = (hsi_data.sum(axis=-1) != 0) & (lidar_data != 0)

    # 提取有效像素
    valid_hsi = hsi_data[valid_mask]
    valid_lidar = lidar_data[valid_mask]

    return valid_hsi, valid_lidar



def check_valid_patch(hsi_patch, lidar_patch):
    """检查patch是否有效"""
    # HSI数据中的无效值(通常为0或NaN)
    if torch.any(torch.isnan(hsi_patch)) or (hsi_patch == 0).all():
        return False

    # LiDAR数据中的无效值
    if torch.isnan(lidar_patch).any() or (lidar_patch == 0).all():
        return False

    return True


# 在数据加载时过滤无效patch
def get_valid_patch(hsi_data, lidar_data, patch_size):
    valid_patches = []
    h, w = hsi_data.shape[:2]

    for i in range(0, h - patch_size + 1):
        for j in range(0, w - patch_size + 1):
            hsi_patch = hsi_data[i:i + patch_size, j:j + patch_size]
            lidar_patch = lidar_data[i:i + patch_size, j:j + patch_size]

            if check_valid_patch(hsi_patch, lidar_patch):
                valid_patches.append((hsi_patch, lidar_patch))

    return valid_patches


if args.is_pretrain == 1:
    model = MAE(
        channel_number=args.channel_num,
        img_size=args.crop_size,
        patch_size=args.patch_size,
        encoder_dim=args.dim,
        encoder_depth=args.depth,
        encoder_heads=args.head,
        decoder_dim=args.dim,
        decoder_depth=args.depth,
        decoder_heads=args.head,
        mask_ratio=args.mask_ratio,
        args=args
    )
else:
    # 进入微调阶段
    model = MAEFinetune(
        channel_number=args.channel_num,
        img_size=args.crop_size,
        patch_size=args.patch_size,
        embed_dim=args.dim,
        depth=args.depth,
        num_heads=args.head,
        num_classes=args.num_classes,
        args=args
    )

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("using " + args.device + " as device")
else:
    print("using cpu as device")
model.to(device)



if __name__ == '__main__':
    # 创建 trainloader 和 testloader
    total_loss = 0
    max_acc = 0

    model = model.to(device)
    model.cuda(device=device)
    optimizer = Optimizer(args.optimizer_name)(
        param=model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        finetune=args.finetune
    )
    # scaler = torch.cuda.amp.GradScaler()
    scaler=torch.amp.GradScaler('cuda')

    # 创建以日期-时间命名的文件夹
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")




    if args.is_pretrain == 1:
        criterion = MSELoss(device=device)
        print("Pretraining!!--------")
        min_loss = 1e8
        batch_iter = 0


        save_dir = os.path.join('model', 'pretrain',current_time)
        os.makedirs(save_dir, exist_ok=True)

        # 初始化日志文件
        log_file = os.path.join(save_dir, 'pretraining_log.txt')


        for epoch in range(args.epoch):
            model.train()
            n = 0
            loss, batch_iter, scaler = Pretrain(args, scaler, model, criterion, optimizer, epoch, batch_iter,
                                                args.batch_size)

            # if loss < min_loss:
            #
            #     state_dict = translate_state_dict(model.state_dict())
            #     state_dict = {
            #         'epoch': epoch,
            #         'state_dict': state_dict,
            #         'optimizer': optimizer.state_dict(),
            #     }
            #     torch.save(
            #         state_dict,
            #         'model/' + 'pretrain_' + args.dataset + '_num' + str(args.pretrain_num) + '_crop_size' + str(
            #             args.crop_size) + '_mask_ratio_' + str(args.mask_ratio) \
            #         + '_DDH_' + str(args.depth) + str(args.dim) + str(args.head) + '_epoch_' + str(
            #             epoch) + '_loss_' + str(loss) + '.pth'
            #     )
            #     min_loss = loss
            with open(log_file, 'a') as f:
                f.write(f"{args.dataset},Num:{args.pretrain_num},Crop_Size:{args.crop_size},Mask_Ratio:{args.mask_ratio},Epoch: {epoch}, Loss: {loss},  Accuracy: {accuracy}\n")

            if loss < min_loss:
                state_dict = translate_state_dict(model.state_dict())
                state_dict = {
                    'epoch': epoch,
                    'state_dict': state_dict,
                    'optimizer': optimizer.state_dict(),
                }
                # 保存模型到指定文件夹，并覆盖之前的模型
                model_path = os.path.join(save_dir, 'best_model.pth')
                torch.save(state_dict, model_path)
                min_loss = loss

    if args.is_train == 1:

        if args.is_load_pretrain==1:
            pretrain_model_path = 'model/train/20250225_223832/best_model.pth'  # 替换为你的预训练模型路径
            checkpoint = torch.load(pretrain_model_path, map_location="cpu")
            model.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded pretrained model from {pretrain_model_path}")

        weights = torch.tensor(list(class_weights.values()), dtype=torch.float32)

        # 将权重移动到 GPU（如果使用 GPU）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = weights.to(device)
        # criterion = nn.CrossEntropyLoss(weight=weights)
        criterion = nn.CrossEntropyLoss()
        batch_iter = 0

        save_dir = os.path.join('model', 'train', current_time)
        os.makedirs(save_dir, exist_ok=True)

        # 初始化日志文件
        log_file = os.path.join(save_dir, 'training_log.txt')


        for epoch in range(args.epoch):
            model.train()
            n = 0
            loss, batch_iter, scaler = Train(args, scaler, model, criterion, optimizer, epoch, batch_iter,
                                             args.batch_size)

            # print(f"all_loader samples: {len(all_loader.dataset)}")
            # print(f"test_loader samples: {len(test_loader.dataset)}")
            # all_labels = [label for _, _, label, _ in all_loader.dataset]
            # print("all_loader labels distribution:", Counter(all_labels))
            #
            # # 检查 test_loader 的标签分布
            # test_labels = [label for _, _, label, _ in test_loader.dataset]
            # print("test_loader labels distribution:", Counter(test_labels))
            #
            #
            # all_batch = next(iter(all_loader))
            # test_batch = next(iter(test_loader))
            #
            # print("all_loader first batch:", all_batch)
            # print("test_loader first batch:", test_batch)




            with open(log_file, 'a') as f:
                f.write(f"{args.dataset},Num:{args.pretrain_num},Crop_Size:{args.crop_size},Mask_Ratio:{args.mask_ratio},Epoch: {epoch}, Loss: {loss},  Accuracy: {accuracy}\n")


            if epoch % args.test_interval == 0:
                # For some datasets (such as Berlin), the test set is too large and the test speed is slow,
                # so it is recommended to split the validation set with a small sample size
                model.eval()
                acc1 = val(args, model)
                print("epoch:", epoch, "acc:", acc1)

                if acc1 > max_acc:
                    state_dict = translate_state_dict(model.state_dict())
                    state_dict = {
                        'epoch': epoch,
                        'state_dict': state_dict,
                        'optimizer': optimizer.state_dict(),
                    }
                    # 保存模型到指定文件夹，并覆盖之前的模型
                    model_path = os.path.join(save_dir, 'best_model.pth')
                    torch.save(state_dict, model_path)
                    max_acc = acc1

    if args.is_test == 1:
        # model_path = 'model/' + 'train_' + args.dataset + '_num' + str(
        #                 args.pretrain_num) + '_crop_size' + str(args.crop_size) + '_mask_ratio_' + str(args.mask_ratio) \
        #                        + '_DDH_' + str(args.depth) + str(args.dim) + str(args.head) + '_epoch_' + str(
        #                 epoch) + '.pth'
        model_path=os.path.join(save_dir, 'best_model.pth')
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        acc1 = val(args, model)

        gt_path = "data/tlse/processed_gt.h5"

        visualize_predictions(args, model, all_loader,gt_path)
