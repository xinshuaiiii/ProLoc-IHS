import torch
from transformers import ViTModel, ViTFeatureExtractor
from PIL import Image
import os
from tqdm import tqdm  # 用于显示进度条
import numpy as np


class ViTFeatureExtractorModel:
    def __init__(self, model_name='google/vit-base-patch16-224-in21k'):
        """
        初始化ViT模型和特征提取器
        :param model_name: Hugging Face transformers中的模型名称
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ViTModel.from_pretrained(model_name)

        # 检查是否有多个GPU
        if torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 个GPU进行并行计算!")
            self.model = torch.nn.DataParallel(self.model)  # 使用 DataParallel 并行计算

        # 将模型移动到GPU
        self.model = self.model.to(self.device)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model.eval()  # 设置模型为评估模式，不会计算梯度

    def preprocess_image(self, image_path):
        """
        预处理图像，将其转换为模型可接受的输入格式
        :param image_path: 图像文件路径
        :return: 处理后的图像张量
        """
        image = Image.open(image_path)
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        return inputs

    def extract_features(self, image_path):
        """
        提取图像的CLS特征嵌入
        :param image_path: 图像文件路径
        """
        inputs = self.preprocess_image(image_path)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}  # 将输入数据移动到GPU

        with torch.no_grad():
            outputs = self.model(**inputs)
        # 获取CLS标记的嵌入向量 (batch_size, hidden_size)
        cls_embedding = outputs.last_hidden_state.cpu().numpy()  # 将输出移回CPU并转换为numpy数组
        return cls_embedding

    def extract_features_from_folder(self, folder_path):
        """
        提取文件夹及其子文件夹中的所有图像的特征嵌入
        :param folder_path: 包含图像的文件夹路径
        :return: 图像路径列表
        """
        assert os.path.exists(folder_path), f"dataset root: {folder_path} does not exist."

        train_images_path = []
        supported = [".jpg", ".JPG", ".png", ".PNG"]

        # 遍历文件夹及其子文件夹中的所有图像
        for root, dirs, files in os.walk(folder_path):
            for img_file in files:
                if os.path.splitext(img_file)[-1] in supported:
                    img_path = os.path.join(root, img_file)
                    train_images_path.append(img_path)

        return train_images_path

    def extract_features_for_all(self, root, save_path="vitembeddings.npy"):
        """
        提取所有图像的CLS特征并保存特征为npy文件
        :param root: 图像文件夹路径
        :param save_path: 保存特征的npy文件路径
        :return: 图像特征
        """
        train_images_path = self.extract_features_from_folder(root)

        # 对所有图像提取特征，并使用进度条显示进度
        train_features = []
        for path in tqdm(train_images_path, desc="Extracting features"):
            feature = self.extract_features(path)
            train_features.append(feature)

        # 将列表转换为NumPy数组
        train_features = np.array(train_features)
        train_features = train_features.squeeze(1)

        # 保存为npy文件
        np.save(save_path, train_features)
        print(f"Features saved to {save_path}")
        print(f"Features shape : {train_features.shape}")

        return train_features
