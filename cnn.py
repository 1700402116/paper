import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

mnist = tf.keras.datasets.cifar10

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#(50000, 32, 32, 3): 50000样本；32x32图像大小；RGB 3通道

#x_train=x_train.reshape(-1,28,28,1)
#x_test=x_test.reshape(-1,28,28,1)   #[None,width,height,channels]
y_train=tf.keras.utils.to_categorical(y_train,num_classes=10)
y_test=tf.keras.utils.to_categorical(y_test,num_classes=10)  #这里是将数字转换为one-hot编码


def AlexNet():
    alexnet = tf.keras.Sequential()

    
    #conv 1 --- pooling 1
    alexnet.add(Conv2D( filters = 32, kernel_size = (3,3), activation = 'relu', input_shape = (32, 32, 3)))
    alexnet.add(MaxPooling2D(pool_size = (2,2))) # overlapping pooling
    
    #conv 2 --- pooling 2
    alexnet.add(Conv2D( filters = 64, kernel_size = (3,3), activation = 'relu'))
    alexnet.add(MaxPooling2D(pool_size = (2,2))) # overlapping pooling
    
    #conv 3 --- pooling 3
    alexnet.add(Conv2D( filters = 128, kernel_size = (3,3), activation = 'relu'))
    alexnet.add(MaxPooling2D(pool_size = (2,2))) # overlapping pooling
    
    #conv 4 --- conv 5
    alexnet.add(Conv2D( filters = 128, kernel_size = (3,3), padding='same', activation = 'relu'))
    alexnet.add(Conv2D( filters = 64, kernel_size = (3,3), padding='same', activation = 'relu'))
    alexnet.add(MaxPooling2D(pool_size = (2,2))) # overlapping pooling

    alexnet.add(Flatten())
    # fc 1-3
    alexnet.add(Dense(64, activation='relu'))
    alexnet.add(Dense(64, activation='relu'))
    alexnet.add(Dense(10, activation='softmax'))

    return alexnet    


#print(x_train.shape)

model = AlexNet()
model.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

log_dir="logs/fit/alexnet_CIFAR10"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x=x_train, y=y_train, epochs=40, validation_data=(x_test, y_test), callbacks=[tensorboard_callback])