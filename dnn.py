import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.cifar10

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
#x_train=tf.keras.utils.to_categorical(x_train,num_classes=100)
#x_train=x_train.reshape(-1,32,32,1)
print(x_train.shape)