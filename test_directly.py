import os
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from model1 import CrossAttentionModel
from vit import ViTFeatureExtractorModel
from prott5 import ProteinEmbeddingExtractor
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, seq_features, attention_masks, img_features, labels=None):
        self.seq_features = seq_features
        self.attention_masks = attention_masks
        self.img_features = img_features
        self.labels = labels

    def __len__(self):
        return len(self.seq_features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.seq_features[idx], self.attention_masks[idx], self.img_features[idx], self.labels[idx]
        return self.seq_features[idx], self.attention_masks[idx], self.img_features[idx]


def sample_based_metrics(all_preds, all_labels):
    n = len(all_preds)
    ATR = np.mean([int(np.array_equal(all_preds[i], all_labels[i])) for i in range(n)])
    Acc = np.mean([len(np.intersect1d(np.where(all_preds[i] == 1)[0], np.where(all_labels[i] == 1)[0])) /
                   len(np.union1d(np.where(all_preds[i] == 1)[0], np.where(all_labels[i] == 1)[0]))
                   if len(np.union1d(np.where(all_preds[i] == 1)[0], np.where(all_labels[i] == 1)[0])) > 0 else 1
                   for i in range(n)])
    Pre = np.mean([len(np.intersect1d(np.where(all_preds[i] == 1)[0], np.where(all_labels[i] == 1)[0])) /
                   len(np.where(all_preds[i] == 1)[0]) if len(np.where(all_preds[i] == 1)[0]) > 0 else 1
                   for i in range(n)])
    Rec = np.mean([len(np.intersect1d(np.where(all_preds[i] == 1)[0], np.where(all_labels[i] == 1)[0])) /
                   len(np.where(all_labels[i] == 1)[0]) if len(np.where(all_labels[i] == 1)[0]) > 0 else 1
                   for i in range(n)])
    F1 = 2 * Pre * Rec / (Pre + Rec) if (Pre + Rec) > 0 else 0
    print(f"Sample-based Metrics:\nATR: {ATR:.4f}\nAcc: {Acc:.4f}\nPre: {Pre:.4f}\nRec: {Rec:.4f}\nF1: {F1:.4f}\n")
    return ATR, Acc, Pre, Rec, F1


def location_based_metrics_micro(all_preds, all_labels):
    TP = np.sum((all_preds == 1) & (all_labels == 1))
    TN = np.sum((all_preds == 0) & (all_labels == 0))
    FP = np.sum((all_preds == 1) & (all_labels == 0))
    FN = np.sum((all_preds == 0) & (all_labels == 1))

    micro_accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    micro_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    micro_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    print(f"Location-based Metrics:\nAcc': {micro_accuracy:.4f}\nPre': {micro_precision:.4f}\nRec': {micro_recall:.4f}\nF1': {micro_f1:.4f}\n")
    return micro_accuracy, micro_precision, micro_recall, micro_f1


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

            outputs = model(
                image_features=img_feat,
                sequence_features=seq_feat,
                attention_mask=attn_mask
            )

            preds = torch.sigmoid(outputs).cpu().numpy()
            preds_thresholded = (preds > thresholds).astype(int)
            all_preds.append(preds_thresholded)

    all_preds = np.vstack(all_preds)

    if has_labels:
        all_labels = np.vstack(all_labels)
        sample_based_metrics(all_preds, all_labels)
        location_based_metrics_micro(all_preds, all_labels)
        return all_preds
    else:
        return all_preds


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    extractor_seq = ProteinEmbeddingExtractor()
    extractor_seq.process_file(args.seq_csv, args.seq_embedding_output, args.seq_mask_output)

    extractor_img = ViTFeatureExtractorModel()
    extractor_img.extract_features_for_all(args.img_folder, args.img_embedding_output)

    seq_features = np.load(args.seq_embedding_output)
    attention_masks = np.load(args.seq_mask_output)
    img_features = np.load(args.img_embedding_output)

    has_labels = os.path.exists(args.label_csv) and args.label_csv.endswith('.csv')
    if has_labels:
        label_df = pd.read_csv(args.label_csv)
        label_columns = [2, 3, 4, 5, 6]
        labels = torch.tensor(label_df.iloc[:, label_columns].values, dtype=torch.float32)
    else:
        labels = None

    seq_features = torch.tensor(seq_features, dtype=torch.float32)
    attention_masks = torch.tensor(attention_masks, dtype=torch.bool)
    img_features = torch.tensor(img_features, dtype=torch.float32)

    dataset = CustomDataset(seq_features, attention_masks, img_features, labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = CrossAttentionModel(
        embedding_dim=512,
        num_heads=256,
        num_layers=6,
        num_classes=5,
        image_embedding_dim=img_features.shape[2],
        sequence_embedding_dim=seq_features.shape[2]
    )
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)

    thresholds = np.array([[float(t) for t in args.thresholds.split(",")]])

    preds = test_model(model, loader, device, has_labels, thresholds)

    if not has_labels:
        pred_df = pd.DataFrame(preds, columns=[f"Label_{i+1}" for i in range(preds.shape[1])])
        pred_df.to_csv(args.pred_output, index=False)
        print(f"Predictions saved to CSV: {args.pred_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_csv', type=str, required=True)
    parser.add_argument('--img_folder', type=str, required=True)
    parser.add_argument('--label_csv', type=str, default="dataset/test.csv")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--seq_embedding_output', type=str, default="seq_test_embeddings.npy")
    parser.add_argument('--seq_mask_output', type=str, default="seq_test_attention_masks.npy")
    parser.add_argument('--img_embedding_output', type=str, default="test_vitembeddings.npy")
    parser.add_argument('--pred_output', type=str, default="predictions.npy")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--thresholds', type=str, default="0.5,0.5,0.5,0.5,0.5")
    args = parser.parse_args()

    main(args)
