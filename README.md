# ProLoc-IHS
ProLoc-IHS is an advanced deep learning model designed to accurately predict the subcellular localization (SCL) of proteins using immunohistochemistry (IHC) images and protein sequences. By combining image features from the Human Protein Atlas and sequence features from UniProt, ProLoc-IHS uses a cross-attention mechanism to capture the complex relationship between protein images and sequence information. It can provide highly accurate multi-label predictions for protein subcellular localization and significantly outperform existing methods.

---
# How to use
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

## Train and test
We provide both training and testing scripts. If you’d like to train the model from scratch, simply run:
```
python train.py
```
If you prefer to use the pretrained features and skip training, you can directly run:
```
python test.py
```
The feature extraction process is integrated into the code—no need for additional preprocessing.

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

