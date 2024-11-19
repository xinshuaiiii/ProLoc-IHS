import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from transformers import T5Tokenizer, T5EncoderModel
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast

class ProteinEmbeddingExtractor:
    def __init__(self, model_path="../prot5", batch_size=1):
        self.batch_size = batch_size
        self.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

        # 加载分词器和模型
        self.tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained(model_path)

        # 将模型移动到当前进程的GPU上
        self.model = self.model.to(self.device)

    def pad_embeddings(self, embeddings, max_len):
        # 填充嵌入序列到 max_len 长度
        return [torch.nn.functional.pad(embedding, (0, 0, 0, max_len - embedding.size(1))) for embedding in embeddings]

    def pad_attention_masks(self, attention_masks, max_len):
        # 填充注意力掩码到 max_len 长度
        return [torch.nn.functional.pad(mask, (0, max_len - mask.size(1))) for mask in attention_masks]

    def process_file(self, filename, output_embeddings_filename, output_masks_filename):
        all_embeddings = []
        all_attention_masks = []

        try:
            df = pd.read_csv(filename)
        except FileNotFoundError:
            print(f"文件 {filename} 未找到，请检查路径。")
            return None, None

        # 假设最后一列包含蛋白质序列
        protein_seq = df.iloc[:, -1].tolist()

        # 替换特定字符，并在每个氨基酸之间添加空格
        sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in protein_seq]

        with tqdm(total=len(sequence_examples), desc=f"Processing {filename}") as pbar:
            for i in range(0, len(sequence_examples), self.batch_size):
                batch_sequences = sequence_examples[i:i + self.batch_size]
                encoding = self.tokenizer.batch_encode_plus(
                    batch_sequences,
                    add_special_tokens=True,
                    padding=True,
                    return_tensors='pt',
                    truncation=True,
                    max_length=6000  # 根据需要调整最大长度
                )

                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)

                with torch.no_grad():
                    with autocast():  # 启用混合精度
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        embeddings = outputs.last_hidden_state  # 形状 (batch_size, seq_len, embedding_size)

                # 保存当前批次的嵌入和注意力掩码
                all_embeddings.append(embeddings.cpu())
                all_attention_masks.append(attention_mask.cpu())

                pbar.update(len(batch_sequences))

        # 找到所有批次中最长的序列长度（最大 seq_len）
        max_len = max([embedding.size(1) for embedding in all_embeddings])

        # 对所有嵌入和掩码进行填充
        all_embeddings = self.pad_embeddings(all_embeddings, max_len)
        all_attention_masks = self.pad_attention_masks(all_attention_masks, max_len)

        # 将所有批次的嵌入和掩码拼接起来
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_attention_masks = torch.cat(all_attention_masks, dim=0)

        # 保存嵌入和注意力掩码
        np.save(output_embeddings_filename, all_embeddings.numpy())
        np.save(output_masks_filename, all_attention_masks.numpy())

        print(f"已保存嵌入到 {output_embeddings_filename}，形状: {all_embeddings.shape}")
        print(f"已保存注意力掩码到 {output_masks_filename}，形状: {all_attention_masks.shape}")

        return output_embeddings_filename, output_masks_filename
