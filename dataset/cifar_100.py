# coding: utf-8
try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path
import gzip
import pickle
import os
import numpy as np
import tarfile
from PIL import Image


url_base = 'https://www.cs.toronto.edu/~kriz/'
file_base = 'cifar-100-python.tar.gz'
extract_base = 'cifar-100-python'

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/" + file_base
extract_path = dataset_dir + "/" + extract_base

train_num = 60000
test_num = 10000
img_dim = (1, 32, 32)
img_size = 1024


def download_cifar_100():
    file_name = file_base
    file_path = save_file

    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")
    headers = {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0"}
    request = urllib.request.Request(url_base+file_name, headers=headers)
    response = urllib.request.urlopen(request).read()
    with open(file_path, mode='wb') as f:
        f.write(response)
    print("Done")


def extract_cifar_100():
    file_path = save_file

    if not os.path.exists(file_path):
        print("File not found. Need download")
        return

    if os.path.exists(extract_path):
        return
    
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=dataset_dir)


def init_cifar_100():
    download_cifar_100()
    extract_cifar_100()

    print("Done!")



def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def _change_one_hot_label(X):
    T = np.zeros((X.size, 100))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


def load_cifar_100(normalize=True, flatten=True, one_hot_label=False):
    """CIFAR-100データセットの読み込み

    Parameters
    ----------
    normalize : 画像のピクセル値を0.0~1.0に正規化する
    one_hot_label :
        one_hot_labelがTrueの場合、ラベルはone-hot配列として返す
        one-hot配列とは、たとえば[0,0,1,0,0,0,0,0,0,0]のような配列
    flatten : 画像を一次元配列に平にするかどうか

    Returns
    -------
    (訓練画像, 訓練ラベル), (テスト画像, テストラベル)
    """
    if not os.path.exists(extract_path):
        init_cifar_100()

    dataset = {}

    train_data = unpickle(extract_path + "/train")
    dataset["train_img"] = train_data[b"data"]
    dataset["train_label"] = np.array(train_data[b"fine_labels"])

    test_data = unpickle(extract_path + "/test")
    dataset["test_img"] = test_data[b"data"]
    dataset["test_label"] = np.array(test_data[b"fine_labels"])



    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 3, 32, 32)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])



if __name__ == '__main__':
    init_cifar_100()
