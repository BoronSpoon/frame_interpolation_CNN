#to force cpu use
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
from tensorflow import keras
from keras.utils import plot_model
from keras.models import load_model
import _pickle as pickle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
#shape = (1060, 2, 128, 128) -convert-> (1060, 128, 128, 2)

with open('train_input.pickle', mode='rb') as f:
    train_input = pickle.load(f)
    train_input = train_input.transpose(0,2,3,1)
with open('train_output.pickle', mode='rb') as f:
    train_output = pickle.load(f)
with open('test_input.pickle', mode='rb') as f:
    test_input = pickle.load(f)
    test_input = test_input.transpose(0,2,3,1)
with open('test_output.pickle', mode='rb') as f:
    test_output = pickle.load(f)
print(np.shape(train_input),np.shape(train_output))

model = Sequential([
    Conv2D(4, kernel_size=(3, 3), padding="same", activation='relu',input_shape =(128, 128, 2),data_format='channels_last'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(8, kernel_size=(3, 3), padding="same", activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(16, kernel_size=(3, 3), padding="same", activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2DTranspose(16, kernel_size=(3, 3), strides=2, padding="same", activation='relu'),
    Conv2DTranspose(4, kernel_size=(3, 3), strides=2, padding="same", activation='relu'),
    Conv2DTranspose(1, kernel_size=(3, 3), strides=2, padding="same", activation='relu'),
])

#saves the model summary into txt file
with open("model.txt", mode='w') as f:
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    str_model_summary = "\n".join(stringlist)
    f.write(str_model_summary)
#saves the model into png file
plot_model(model, to_file='model.png')

sgd = SGD(lr = 0.00000001)

model.compile(loss='mse',
              optimizer=sgd,
              metrics=['accuracy'],)

import os
import os.path

PATH='./my_model.h5'

if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
    model = load_model('my_model.h5')

history = model.fit(train_input, train_output,
                    #initial_epoch = 299,
                    epochs=40000,
                    batch_size=64,
                    validation_data=(test_input,test_output),
                    verbose = 2)

test_loss, test_acc = model.evaluate(test_input, test_output)

print('Test loss:', test_loss, 'Test accuracy:', test_acc)

model.save('my_model.h5')

#test phase
test_indices = 100
test_input_1 = test_input[test_indices,:,:,0]
test_input_2 = test_input[test_indices,:,:,1]
test_output = model.predict(np.array([test_input[test_indices,:]]))
plt.subplot(131)
plt.imshow(test_input_1)
plt.subplot(132)
plt.imshow(test_output[0,:,:,0])
plt.subplot(133)
plt.imshow(test_input_2)
plt.savefig('test.png', pad_inches=0.1) #pad_inches is used to prevent the output png from having excess padding
plt.show()
plt.cla()
plt.clf()

#plots the loss and val_loss
plt.xlabel("epochs")
plt.ylabel("loss")
plt.plot(history.history['loss'],label="loss")
plt.plot(history.history['val_loss'],label="val_loss")
plt.legend(loc = 'upper right')
plt.title('loss and validation_loss')

#saves the plot into png and displays the plot
plt.savefig('train1.png', pad_inches=0.1) #pad_inches is used to prevent the output png from having excess padding
plt.show()