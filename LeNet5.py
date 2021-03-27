import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# x_train = np.expand_dims(x_train, axis=3)
# x_test = np.expand_dims(x_test, axis=3)

#如果不是minst数据集，注释掉下两行
x_train=x_train.reshape(-1,28,28,1)
x_test=x_test.reshape(-1,28,28,1)   #[None,width,height,channels]

y_train=tf.keras.utils.to_categorical(y_train,num_classes=10)
y_test=tf.keras.utils.to_categorical(y_test,num_classes=10)  #这里是将数字转换为one-hot编码


def LeNet_5():
    lenet = tf.keras.models.Sequential()
    lenet.add(Conv2D(filters=6, kernel_size=(5,5), padding='valid', input_shape=(28,28,1), activation='tanh'))
    lenet.add(MaxPooling2D(pool_size=(2,2)))
    lenet.add(Conv2D(filters=16, kernel_size=(5,5), padding='valid', activation='tanh'))
    lenet.add(MaxPooling2D(pool_size=(2,2)))
    lenet.add(Flatten())
    lenet.add(Dense(120, activation='tanh'))
    lenet.add(Dense(84, activation='tanh'))
    lenet.add(Dense(10, activation='softmax'))
    return lenet    


#print(x_train.shape)

model = LeNet_5()
sgd = tf.keras.optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

log_dir="logs/fit/LeNet-5-mnist"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x=x_train, 
        y=y_train, 
        epochs=100, 
        validation_data=(x_test, y_test), 
        callbacks=[tensorboard_callback])

