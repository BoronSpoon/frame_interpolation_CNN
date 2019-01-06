#to force cpu use
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.models import load_model
from tensorflow import keras
from matplotlib import pyplot as plt
import _pickle as pickle
import numpy as np

with open('test_input.pickle', mode='rb') as f:
    test_input = pickle.load(f)
    test_input = test_input.transpose(0,2,3,1)
with open('test_output.pickle', mode='rb') as f:
    test_output = pickle.load(f)

model = load_model('my_model.h5')

#test phase
test_indices = 200
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