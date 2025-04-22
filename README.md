# ProLoc-IHS

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
