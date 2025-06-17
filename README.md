# ProLoc-IHS
ProLoc-IHS is an advanced deep learning model designed to accurately predict the subcellular localization (SCL) of proteins using immunohistochemistry (IHC) images and protein sequences. By combining image features from the Human Protein Atlas and sequence features from UniProt, ProLoc-IHS uses a cross-attention mechanism to capture the complex relationship between protein images and sequence information. It can provide highly accurate multi-label predictions for protein subcellular localization and significantly outperform existing methods.

---
# How to use
## Creating a Virtual Environment
This section describes how to use the trained ProLoc-IHS model to make subcellular localization predictions using a CSV file of sequences and a folder of IHC images.
To run the code, we need to create a virtual environment using Anaconda, and install the required dependencies.The command is as follows：
```
git clone https://github.com/xinshuaiiii/ProLoc-IHS.git
conda create -n ProLoc-IHS pyhton=3.7.13
conda activate ProLoc-IHS
pip install -r requirements.txt
```

To download images, run:
```bash
python download.py
```
Case 1: If you have a complete test set and labels to evaluate the indicators
```
python test_directly.py \
  --seq_csv test_name_URL.csv \
  --img_folder test \
  --label_csv dataset/test.csv \
```
This will extract features, run inference, and compute evaluation metrics.


Case 2: If you just want to predict the location
```
python test_directly.py \
  --seq_csv test_name_URL.csv \
  --img_folder test \
```
This will skip metric computation and only output the predicted results.

# Citation
If you use this work in your research, please cite the following paper:
If you have any questions, please contact Yun Liu(liuyun313@jlu.edu.cn) or Shuai Xin(xinshuai23@mails.jlu.edu.cn)

<pre>
@article{liu2025proloc,
  title={ProLoc-IHS: Multi-label protein subcellular localization based on immunohistochemical images and sequence information},
  author={Liu, Fu and Xin, Shuai and Liu, Yun},
  journal={International Journal of Biological Macromolecules},
  pages={144096},
  year={2025}
}
</pre>

