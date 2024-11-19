import torch
import torch.nn as nn
class CrossAttentionModel(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, image_embedding_dim, sequence_embedding_dim,num_classes, dropout=0.5):
        """
        初始化模型
        :param embedding_dim: 输入特征的嵌入维度
        :param num_heads: 多头注意力的头数
        :param num_layers: 重复的层数
        :param num_classes: 分类层的输出类别数
        :param dropout: dropout的概率
        """
        super(CrossAttentionModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.image_projection = nn.Linear(image_embedding_dim, embedding_dim)
        self.sequence_projection = nn.Linear(sequence_embedding_dim, embedding_dim)

        # 定义交叉注意力层和自注意力层
        self.cross_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout)
        self.self_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout)

        # 前馈神经网络
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embedding_dim, embedding_dim)
        )

        # 残差连接和层归一化
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        # 分类层
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, image_features, sequence_features, attention_mask=None):
        """
        前向传播
        :param image_features: 图像特征，形状为 (batch_size, patch, embedding_dim)
        :param sequence_features: 序列特征，形状为 (batch_size, seq_len, embedding_dim)
        :param attention_mask: 注意力掩码，形状为 (batch_size, seq_len)
        :return: 分类结果
        """
        batch_size, seq_len, _ = sequence_features.size()
        image_features = self.image_projection(image_features)
        sequence_features = self.sequence_projection(sequence_features)
        sequence_features = sequence_features.permute(1, 0, 2)  # (seq_len, batch_size, embedding_dim)
        image_features = image_features.permute(1, 0, 2)        # (patch, batch_size, embedding_dim)
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0) # (seq_len, batch_size)
        else:
            key_padding_mask = None
        # 交叉注意力层
        attn_output, _ = self.cross_attention(query=image_features, key=sequence_features, value=sequence_features,)
        # 重复N次
        for _ in range(self.num_layers):
            # 自注意力层
            self_attn_output, _ = self.self_attention(query=attn_output, key=attn_output, value=attn_output)
            # 残差连接和层归一化
            attn_output = self.layer_norm1(attn_output + self_attn_output)
            # 前馈神经网络
            ff_output = self.feed_forward(attn_output)
            # 残差连接和层归一化
            attn_output = self.layer_norm2(attn_output + ff_output)

        attn_output = attn_output.permute(1, 0, 2)

        # 平均池化后进行分类
        pooled_output = attn_output.mean(dim=1)  # (batch_size, embedding_dim)
        logits = self.classifier(pooled_output)  # (batch_size, num_classes)

        return logits
