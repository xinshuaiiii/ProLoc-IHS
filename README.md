# ProLoc-IHS
ProLoc-IHS is an advanced deep learning model designed to accurately predict the subcellular localization (SCL) of proteins using immunohistochemistry (IHC) images and protein sequences. By combining image features from the Human Protein Atlas and sequence features from UniProt, ProLoc-IHS uses a cross-attention mechanism to capture the complex relationship between protein images and sequence information. It can provide highly accurate multi-label predictions for protein subcellular localization and significantly outperform existing methods.

---
# How to use
This section describes how to use the trained ProLoc-IHS model to make subcellular localization predictions using a CSV file of sequences and a folder of IHC images.
## Create a Virtual Environment
To run the code, we need to create a virtual environment using Anaconda, and install the required dependencies. The command is as follows：
```
git clone https://github.com/xinshuaiiii/ProLoc-IHS.git
conda create -n ProLoc-IHS pyhton=3.7.13
conda activate ProLoc-IHS
pip install -r requirements.txt
```
## Prepare your data
Proteins IHC images and sequences are necessary to perform ProLoc-IHS. IHC images should be of what format, and sequences should be of what format.

Attention: your IHC images and sequences should be in same order, or your will get wrong results.

## Predict
```
python test_directly.py \
  --seq_csv sequence.csv \
  --img_folder ihcFolder \
```
This will generate the prediction results in xxx.file, in which xxx.


# Citation
If you use this work in your research, please cite the following paper.

<pre>
@article{liu2025proloc,
  title={ProLoc-IHS: Multi-label protein subcellular localization based on immunohistochemical images and sequence information},
  author={Liu, Fu and Xin, Shuai and Liu, Yun},
  journal={International Journal of Biological Macromolecules},
  pages={144096},
  year={2025}
}
</pre>


If you have any questions, please contact Yun Liu(liuyun313@jlu.edu.cn) or Shuai Xin(xinshuai23@mails.jlu.edu.cn)

