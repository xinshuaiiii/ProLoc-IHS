# ProLoc-IHS
This model is a multi-label protein subcellular localization model that integrates immunohistochemical (IHC) images and protein sequence information. It is designed to address the challenge of predicting proteins that localize to multiple subcellular compartments, especially when image and sequence modalities provide heterogeneous but complementary cues.

The model employs a cross-attention mechanism to map semantic relationships between modalities and enhances discriminative learning through a fusion framework. This enables accurate localization prediction across complex spatial distributions.

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
Once the dataset files are prepared, you can directly run the following command to train and generate subcellular localization predictions. The code automatically handles feature extraction internally, so no additional preprocessing is required:
```
python train.py
```
and
```
python test.py
```

# Citation
If you use this work in your research, please cite the following paper:

<pre>
```bibtex
@article{liu2025proloc,
  title={ProLoc-IHS: Multi-label protein subcellular localization based on immunohistochemical images and sequence information},
  author={Liu, Fu and Xin, Shuai and Liu, Yun},
  journal={International Journal of Biological Macromolecules},
  pages={144096},
  year={2025}
}
```
</pre>

