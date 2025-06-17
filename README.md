# ProLoc-IHS
ProLoc-IHS is an advanced deep learning model designed to accurately predict the subcellular localization (SCL) of proteins using immunohistochemistry (IHC) images and protein sequences. By combining image features from the Human Protein Atlas and sequence features from UniProt, ProLoc-IHS uses a cross-attention mechanism to capture the complex relationship between protein images and sequence information. It can provide highly accurate multi-label predictions for protein subcellular localization and significantly outperform existing methods.

---
# How to train from scratch
## Creating a Virtual Environment
To run the code, we need to create a virtual environment using Anaconda, and install the required dependencies.The command is as follows：
```
conda create -n ProLoc-IHS pyhton=3.7.13
conda activate ProLoc-IHS
pip install -r requirements.txt
```

## Dataset Preparation

In the `dataset/` folder, there are four CSV files:

- **`train.csv`** and **`test.csv`**: Contain labels, sequences, and metadata for training and testing.
- **`train_img_URL.csv`** and **`test_img_URL.csv`**: Contain image URLs used to download IHC images.

To download images, run:
```bash
python download.py
```

## Train
The feature extraction process is integrated into the code—no need for additional preprocessing. Run:
```
python train.py
```

In addition to retraining, we also provide pre-trained weights for direct prediction.
# How to use
This section describes how to use the trained ProLoc-IHS model to make subcellular localization predictions using a CSV file of sequences and a folder of IHC images.

Predict and Evaluate (with Ground Truth Labels)：
```
python test_directly.py \
  --seq_csv test_name_URL.csv \
  --img_folder test \
  --label_csv dataset/test.csv \
  --model_path best_model.pth
```
This will extract features, run inference, and compute evaluation metrics.


Predict Only (No Ground Truth):
```
python predict.py \
  --seq_csv test_name_URL.csv \
  --img_folder test \
  --label_csv "" \
  --model_path best_model.pth
```
This will skip metric computation and only output the predicted results.

# Citation
If you use this work in your research, please cite the following paper:

<pre>
@article{liu2025proloc,
  title={ProLoc-IHS: Multi-label protein subcellular localization based on immunohistochemical images and sequence information},
  author={Liu, Fu and Xin, Shuai and Liu, Yun},
  journal={International Journal of Biological Macromolecules},
  pages={144096},
  year={2025}
}
</pre>

