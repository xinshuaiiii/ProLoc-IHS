# ProLoc-IHS
We propose a novel IHC image protein SCL prediction model **ProLoc-IHS** that combines sequence features. First, a bimodal dataset containing IHC images and protein sequences from the Human Protein Atlas (HPA) and UniProt is compiled. Then, ProLoc-IHS extracts embeddings from IHC images and protein sequences using the visual language model Vision Transformer (ViT) and the protein language model ProtT5, respectively. These embeddings are fused using a cross-attention module, and the fused features are input into the feature learning module of ProLoc-IHS, which contains a multi-head attention mechanism, a feedforward neural network, and a residual connection.

Finally, the binary cross entropy (BCE) and the Focal loss function are introduced into the feature learning module to solve the multi-label classification task. Experimental results show that ProLoc-IHS outperforms other prediction models.

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
@article{liu2025proloc,
  title={ProLoc-IHS: Multi-label protein subcellular localization based on immunohistochemical images and sequence information},
  author={Liu, Fu and Xin, Shuai and Liu, Yun},
  journal={International Journal of Biological Macromolecules},
  pages={144096},
  year={2025}
}
</pre>

