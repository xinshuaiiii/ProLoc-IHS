from prott5 import ProteinEmbeddingExtractor
from vit import ViTFeatureExtractorModel
extractor_seq = ProteinEmbeddingExtractor()
extractor_seq.process_file("train_name_URL.csv","seq_train_embeddings.npy","seq_train_attention_masks.npy")
extractor_img = ViTFeatureExtractorModel()
extractor_img.extract_features_for_all( "train"  ,"train_vitembeddings.npy")
