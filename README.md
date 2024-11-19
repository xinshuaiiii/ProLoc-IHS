# ProLoc-IHS

##Requirements
First create a virtual environment
'''
conda create -n ProLoc-IHS python3.10
conda activate ProLoc-IHS
pip install -r requirements.txt
'''


##Dataset

In the _dataset_ folder, there are four csv files, where _train/test.csv_ is the training set and test set information, including labels, sequences and other information, and _train/test_img_URL.csv_ contains the URL address of the image information. You need to use `python download.py `to download the IHC image.
