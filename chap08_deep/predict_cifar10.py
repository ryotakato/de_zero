import sys, os
sys.path.append(os.pardir)
import numpy as np
from matplotlib.image import imread
from deep_convnet_for_cifar10 import DeepConvNet
from PIL import Image


# meta data load
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

meta = unpickle("../dataset/cifar-10-batches-py/batches.meta")
label_names = meta[b'label_names']


# load
network = DeepConvNet(input_dim=(3, 32, 32))
network.load_params("deep_convnet_params_cifar10.pkl")
print("Loaded Network Parameters!")



print("-- file_name: prediction_result --------------------")


img_dir = "../dataset/img/"
files = os.listdir(img_dir)
for f in files:
    with Image.open(img_dir + f) as img:
    
        # resize
        img = img.resize((32, 32))
    
        # convert
        array_img = np.array(img)
        array_img = array_img.transpose(2, 0, 1)
        array_img = np.array([array_img])
        # normalize
        array_img = array_img.astype(np.float32)
        array_img = array_img / 255.0
    
        # predict
        ans = network.predict(array_img, train_flg=False)
    
        #print(np.argmax(ans))
        predict_label = label_names[np.argmax(ans)].decode('utf-8')
        print(f + ": " + predict_label)




