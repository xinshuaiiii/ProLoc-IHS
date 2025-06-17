

**ProLoc-IHS** is a multi-label protein subcellular localization model that integrates immunohistochemical (IHC) images and protein sequence information. It is designed to address the challenge of predicting proteins that localize to multiple subcellular compartments, especially when image and sequence modalities provide heterogeneous but complementary cues.

The model employs a cross-attention mechanism to map semantic relationships between modalities and enhances discriminative learning through a fusion framework. This enables accurate localization prediction across complex spatial distributions.

---
# Creating a Virtual Environment
To run the code, we need to create a virtual environment using Anaconda, and install the required dependencies.The command is as follows：
```
conda create -n ProLoc-IHS pyhton=3.7.13
conda activate ProLoc-IHS
pip install -r requirements.txt
```


# Dataset
In the _dataset_ folder, there are four csv files, where _train/test.csv_ is the training set and test set information, including labels, sequences and other information, and _train/test_img_URL.csv_ contains the URL address of the image information. You need to use `python download.py `to download the IHC image.

# Train and test
After preparing the features and training files,run:
```
python train.py
```
and
```
python test.py
```
