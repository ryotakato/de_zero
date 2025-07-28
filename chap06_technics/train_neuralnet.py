import sys, os, time
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
from multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import *


#start = time.perf_counter()
start = time.time()

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)


# parameter
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100


train_loss_list = []
train_acc_list = []
test_acc_list = []
# iterate number per each epoch (epoch = train_size / batch_size)
iter_per_epoch = max(train_size / batch_size, 1)


networks = {"chap05+SGD": {"network":TwoLayerNet(input_size=784, hidden_size=50, output_size=10), "optimizer": SGD()},
            "chap05+AdaGrad": {"network":TwoLayerNet(input_size=784, hidden_size=50, output_size=10), "optimizer": AdaGrad()},
            "chap06+AdaGrad": {"network":MultiLayerNetExtend(input_size=784, hidden_size_list=[50], output_size=10, weight_init_std=0.01), "optimizer": AdaGrad()},
            "chap06+AdaGrad+weight init he:": {"network":MultiLayerNetExtend(input_size=784, hidden_size_list=[50], output_size=10, weight_init_std="relu"), "optimizer": AdaGrad()},
            "chap06+AdaGrad+weight init xavier:": {"network":MultiLayerNetExtend(input_size=784, hidden_size_list=[50], output_size=10, weight_init_std="sigmoid"), "optimizer": AdaGrad()},
            "chap06+AdaGrad+weight init he+dropout:": {"network":MultiLayerNetExtend(input_size=784, hidden_size_list=[50], output_size=10, weight_init_std="sigmoid", use_dropout=True, use_batchnorm=False), "optimizer": AdaGrad()},
            "chap06+AdaGrad+weight init he+batchnorm:": {"network":MultiLayerNetExtend(input_size=784, hidden_size_list=[50], output_size=10, weight_init_std="sigmoid", use_dropout=False, use_batchnorm=True), "optimizer": AdaGrad()},
            "chap06+AdaGrad+weight init he+dropout+batchnorm:": {"network":MultiLayerNetExtend(input_size=784, hidden_size_list=[50], output_size=10, weight_init_std="sigmoid", use_dropout=True, use_batchnorm=True), "optimizer": AdaGrad()},
            "chap06+AdaGrad+weight init he+dropout+hidden100,50:": {"network":MultiLayerNetExtend(input_size=784, hidden_size_list=[100,50], output_size=10, weight_init_std="sigmoid", use_dropout=True, use_batchnorm=False), "optimizer": AdaGrad()},
            "chap06+AdaGrad+weight init he+batchnorm+hidden100,50:": {"network":MultiLayerNetExtend(input_size=784, hidden_size_list=[100,50], output_size=10, weight_init_std="sigmoid", use_dropout=False, use_batchnorm=True), "optimizer": AdaGrad()},
            "chap06+AdaGrad+weight init he+batchnorm+hidden100,50+hyper para:": {"network":MultiLayerNetExtend(input_size=784, hidden_size_list=[100,50], output_size=10, weight_init_std="sigmoid", use_dropout=False, use_batchnorm=True, weight_decay_lambda=3.86e-7), "optimizer": AdaGrad(lr=0.0092)},
            }

for key, classes in networks.items():

    network = classes["network"]
    optimizer = classes["optimizer"]
    
    for i in range(iters_num):
        #print(f'Iter Num: {i}')
        # mini batch
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
    
        # calculate gradient by backpropagation
        grads = network.gradient(x_batch, t_batch)
    
        # update parameter
        optimizer.update(network.params, grads)
    
        # record learning progress
        if type(network) is MultiLayerNetExtend:
            loss = network.loss(x_batch, t_batch, train_flg=True)
        else:
            loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
    
    
        # calculate accuracy per epoch
        #if i % iter_per_epoch == 0:
        #    train_acc = network.accuracy(x_train, t_train)
        #    test_acc = network.accuracy(x_test, t_test)
        #    train_acc_list.append(train_acc)
        #    test_acc_list.append(test_acc)
        #    print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
    
    train_acc = network.accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)
    print(f'{key} | train acc, test acc | {train_acc}, {test_acc}')
    
    #end = time.perf_counter()
    #end = time.time()
    #print(f'time : {end-start} second')
    
    # show loss 
    #x = np.arange(len(train_loss_list)) 
    #plt.plot(x, train_loss_list)
    #plt.ylim(0, 2.5)
    #plt.show()
    #
    ## show accuracy
    #markers = {'train': 'o', 'test': 's'}
    #x = np.arange(len(train_acc_list))
    #plt.plot(x, train_acc_list, label='train acc')
    #plt.plot(x, test_acc_list, label='test acc', linestyle='--')
    #plt.xlabel("epochs")
    #plt.ylabel("accuracy")
    #plt.ylim(0, 1.0)
    #plt.legend(loc="lower right")
    #plt.show()



