import sys, os, time
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.cifar_10 import load_cifar_10
from deep_convnet_for_cifar10 import DeepConvNet
from common.trainer import Trainer

start = time.time()

# load data
(x_train, t_train), (x_test, t_test) = load_cifar_10(flatten=False)

max_epochs = 25

network = DeepConvNet(input_dim=(3, 32, 32))

trainer = Trainer(network, x_train, t_train, x_test, t_test, epochs=max_epochs, mini_batch_size=100, optimizer="Adam", optimizer_param={'lr':0.001}, evaluate_sample_num_per_epoch=1000, verbose_iter=False)

trainer.train()

# save
network.save_params("deep_convnet_params_cifar10.pkl")
print("Saved Network Parameters!")

end = time.time()
print(f'time : {end-start} second')

markers = {'train': 'o', 'test': 's'}
x = np.arange(len(trainer.train_acc_list))
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

