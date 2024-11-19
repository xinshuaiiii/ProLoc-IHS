import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from model import CrossAttentionModel

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

# 多任务损失函数：主任务使用 BCE，次任务使用 Focal 和 BCE 的结合
def multi_task_loss(outputs, labels, focal_weight=4, bce_weight=4, main_loss_weight=3, secondary_loss_weight=10):
    # 主任务：BCEWithLogitsLoss 对所有标签
    main_loss = nn.functional.binary_cross_entropy_with_logits(outputs, labels)

    # 次任务：专门针对第5个标签（索引为4），结合 Focal Loss 和 BCEWithLogitsLoss
    focal_loss = focal_criterion(outputs[:, 4], labels[:, 4])
    bce_loss = nn.functional.binary_cross_entropy_with_logits(outputs[:, 4], labels[:, 4])
    combined_loss = focal_weight * focal_loss + bce_weight * bce_loss

    # 最终损失：主任务损失 + 次任务损失
    total_loss = main_loss_weight * main_loss + secondary_loss_weight * combined_loss
    return total_loss

# 定义一个自定义数据集
class CustomDataset(Dataset):
    def __init__(self, seq_features, attention_masks, img_features, labels):
        self.seq_features = seq_features
        self.attention_masks = attention_masks
        self.img_features = img_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq_feat = self.seq_features[idx]
        attn_mask = self.attention_masks[idx]
        img_feat = self.img_features[idx]
        label = self.labels[idx]
        return seq_feat, attn_mask, img_feat, label

# 主函数
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载特征和标签
    print("加载序列特征")
    seq_features = np.load("seq_train_embeddings.npy")
    print("加载序列掩码")
    attention_masks = np.load("seq_train_attention_masks.npy")
    print("加载图像特征")
    img_features = np.load("vitembeddings.npy")

    label_df = pd.read_csv("train.csv")
    label_columns = [2, 3, 4, 5, 6]
    labels = label_df.iloc[:, label_columns].values
    labels = torch.tensor(labels, dtype=torch.float32)

    # 保证样本数一致
    num_samples = labels.shape[0]
    seq_features = seq_features[:num_samples]
    attention_masks = attention_masks[:num_samples]
    img_features = img_features[:num_samples]

    pos_weight = torch.tensor(np.sum(labels.numpy() == 0, axis=0) / np.sum(labels.numpy() == 1, axis=0), dtype=torch.float32).to(device)
    print(f"类别不平衡权重：{pos_weight}")

    # 将 NumPy 数组转换为 PyTorch 张量
    seq_features = torch.tensor(seq_features, dtype=torch.float32)
    attention_masks = torch.tensor(attention_masks, dtype=torch.bool)
    img_features = torch.tensor(img_features, dtype=torch.float32)

    # 创建训练集的数据集对象
    train_dataset = CustomDataset(
        seq_features, attention_masks, img_features, labels
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # 定义模型、损失函数和优化器
    embedding_dim = 512
    num_heads = 8
    num_layers = 6
    num_classes = labels.shape[1]
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
    model = model.to(device)

    # 定义 FocalLoss 和 BCEWithLogitsLoss
    focal_criterion = FocalLoss(alpha=1, gamma=2)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # 训练模型并保存损失最小的模型
    num_epochs = 100
    best_loss = float('inf')  # 初始化最佳损失为无穷大
    best_model_path = "best_model.pth"

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # 在训练循环中添加 tqdm 进度条
        for batch in tqdm(train_loader, desc="Training", leave=False):
            seq_feat, attn_mask, img_feat, label = batch
            seq_feat = seq_feat.to(device)
            attn_mask = attn_mask.to(device)
            img_feat = img_feat.to(device)
            label = label.to(device)

            # 前向传播
            outputs = model(
                image_features=img_feat,
                sequence_features=seq_feat,
                attention_mask=attn_mask
            )

            # 计算多任务损失
            loss = multi_task_loss(outputs, label)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * seq_feat.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # 保存损失最小的模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with Loss: {best_loss:.4f} at epoch {epoch + 1}")

    print("训练完成！")
