import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score
from model1 import CrossAttentionModel  # 引入模型类
from tqdm import tqdm  # 导入 tqdm 用于显示进度条
import pandas as pd


# 定义一个自定义数据集
class CustomDataset(Dataset):
    def __init__(self, seq_features, attention_masks, img_features, labels=None):
        self.seq_features = seq_features
        self.attention_masks = attention_masks
        self.img_features = img_features
        self.labels = labels

    def __len__(self):
        return len(self.seq_features)

    def __getitem__(self, idx):
        seq_feat = self.seq_features[idx]
        attn_mask = self.attention_masks[idx]
        img_feat = self.img_features[idx]
        if self.labels is not None:
            label = self.labels[idx]
            return seq_feat, attn_mask, img_feat, label
        else:
            return seq_feat, attn_mask, img_feat


# 基于样本的度量计算函数
def sample_based_metrics(all_preds, all_labels):
    n = len(all_preds)
    m = all_preds.shape[1]

    # ATR (Absolute True Rate)
    ATR = np.mean([int(np.array_equal(all_preds[i], all_labels[i])) for i in range(n)])

    # Acc (Accuracy)
    Acc = np.mean([len(np.intersect1d(np.where(all_preds[i] == 1)[0], np.where(all_labels[i] == 1)[0])) /
                   len(np.union1d(np.where(all_preds[i] == 1)[0], np.where(all_labels[i] == 1)[0]))
                   if len(np.union1d(np.where(all_preds[i] == 1)[0], np.where(all_labels[i] == 1)[0])) > 0 else 1
                   for i in range(n)])

    # Pre (Precision)
    Pre = np.mean([len(np.intersect1d(np.where(all_preds[i] == 1)[0], np.where(all_labels[i] == 1)[0])) /
                   len(np.where(all_preds[i] == 1)[0]) if len(np.where(all_preds[i] == 1)[0]) > 0 else 1
                   for i in range(n)])

    # Rec (Recall)
    Rec = np.mean([len(np.intersect1d(np.where(all_preds[i] == 1)[0], np.where(all_labels[i] == 1)[0])) /
                   len(np.where(all_labels[i] == 1)[0]) if len(np.where(all_labels[i] == 1)[0]) > 0 else 1
                   for i in range(n)])

    # F1 (F1 Score)
    F1 = 2 * Pre * Rec / (Pre + Rec) if (Pre + Rec) > 0 else 0

    print(f"Sample-based Metrics:\nATR: {ATR:.4f}\nAcc: {Acc:.4f}\nPre: {Pre:.4f}\nRec: {Rec:.4f}\nF1: {F1:.4f}\n")
    return ATR, Acc, Pre, Rec, F1


# 基于位置的微观度量计算函数
def location_based_metrics_micro(all_preds, all_labels):
    # 汇总所有标签的 TP, TN, FP, FN
    TP = np.sum((all_preds == 1) & (all_labels == 1))
    TN = np.sum((all_preds == 0) & (all_labels == 0))
    FP = np.sum((all_preds == 1) & (all_labels == 0))
    FN = np.sum((all_preds == 0) & (all_labels == 1))

    # 微观 (micro) 度量计算
    micro_accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    micro_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    micro_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    print(f"Location-based Metrics:\nAcc': {micro_accuracy:.4f}\nPre': {micro_precision:.4f}\nRec': {micro_recall:.4f}\nF1': {micro_f1:.4f}\n")
    return micro_accuracy, micro_precision, micro_recall, micro_f1


# 测试函数，增加整体准确率的计算
def test_model(model, test_loader, device, has_labels, thresholds):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            if has_labels:
                seq_feat, attn_mask, img_feat, labels = batch
                labels = labels.to(device)
                all_labels.append(labels.cpu().numpy())
            else:
                seq_feat, attn_mask, img_feat = batch

            seq_feat = seq_feat.to(device)
            attn_mask = attn_mask.to(device)
            img_feat = img_feat.to(device)

            # 前向传播，获取模型输出
            outputs = model(
                image_features=img_feat,
                sequence_features=seq_feat,
                attention_mask=attn_mask
            )

            # 使用最佳阈值对预测结果进行二进制化
            preds = torch.sigmoid(outputs).cpu().numpy()
            preds_thresholded = (preds > thresholds).astype(int)  # 使用每个标签的最佳阈值
            all_preds.append(preds_thresholded)

    all_preds = np.vstack(all_preds)

    if has_labels:
        all_labels = np.vstack(all_labels)

        # 基于样本的度量
        ATR, Acc, Pre, Rec, F1 = sample_based_metrics(all_preds, all_labels)

        # 基于位置的微观度量
        Acc_prime_micro, Pre_prime_micro, Rec_prime_micro, F1_prime_micro = location_based_metrics_micro(all_preds, all_labels)

        return ATR, Acc, Pre, Rec, F1, Acc_prime_micro, Pre_prime_micro, Rec_prime_micro, F1_prime_micro
    else:
        return all_preds


# 主函数
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载测试集特征
    print("加载测试集序列特征")
    seq_features = np.load("seq_test_embeddings.npy")
    print("加载测试集序列掩码")
    attention_masks = np.load("seq_test_attention_masks.npy")
    print("加载测试集图像特征")
    img_features = np.load("test_vitembeddings.npy")

    # 加载模型
    embedding_dim = 512
    num_heads = 256
    # num_heads = 8
    num_layers = 6
    num_classes = 5  # 假设测试集中有 5 个标签类别
    image_embedding_dim = img_features.shape[2]
    sequence_embedding_dim = seq_features.shape[2]

    model = CrossAttentionModel(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_classes=num_classes,
        image_embedding_dim=image_embedding_dim,
        sequence_embedding_dim=sequence_embedding_dim
    )

    # 加载保存的模型权重
    best_model_path = "best_model_heads_256.pth"
    model.load_state_dict(torch.load(best_model_path))
    model = model.to(device)

    # 手动定义最佳阈值
    # thresholds = np.array([[0.34 ,0.38 ,0.4 , 0.56, 0.48]])
    thresholds = np.array([[0.5,0.5,0.5,0.5,0.5]])

    # 如果有测试集标签，可以加载标签数据
    try:
        label_df = pd.read_csv("test_name_URL.csv")
        label_columns = [2, 3, 4, 5, 6]  # 对应的标签列
        labels = label_df.iloc[:, label_columns].values
        labels = torch.tensor(labels, dtype=torch.float32)
        has_labels = True
    except FileNotFoundError:
        print("没有找到标签文件，将不会计算测试集的 F1 分数。")
        labels = None
        has_labels = False

    # 将 NumPy 数组转换为 PyTorch 张量
    seq_features = torch.tensor(seq_features, dtype=torch.float32)
    attention_masks = torch.tensor(attention_masks, dtype=torch.bool)
    img_features = torch.tensor(img_features, dtype=torch.float32)

    # 创建测试集的数据集对象
    if has_labels:
        test_dataset = CustomDataset(seq_features, attention_masks, img_features, labels)
    else:
        test_dataset = CustomDataset(seq_features, attention_masks, img_features)

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 在测试集上测试模型
    test_model(model, test_loader, device, has_labels, thresholds)
