import os
import glob
from PIL import Image
import numpy as np
import _pickle as pickle
import random

path_list = [[] for i in range(20)]
for i in range(20):
    path_list[i].extend(glob.glob('**/obj' + str(i+1) + '_*.png'))

img = [[np.asarray(Image.open(j).convert('L')) for j in i] for i in path_list]
img_size = img[0][0].shape[0]

train_input = []
train_output = []
test_input = []
test_output = []

for i in img:
    test_size = int((len(i) - 2)/4)
    test_indices = random.sample(list(range(len(i)-2)), test_size)
    for j in range(len(i)-2):
        if j in test_indices:
            test_input.append(np.array([i[j],i[j + 2]]))
            test_output.append(i[j + 1])
        else:
            train_input.append(np.array([i[j],i[j + 2]]))
            train_output.append(i[j+1])

with open('train_input.pickle', mode='wb') as f:
    pickle.dump(np.array(train_input), f)
with open('train_output.pickle', mode='wb') as f:
    pickle.dump(np.array(train_output)[:,:,:,np.newaxis], f)
with open('test_input.pickle', mode='wb') as f:
    pickle.dump(np.array(test_input), f)
with open('test_output.pickle', mode='wb') as f:
    pickle.dump(np.array(test_output)[:,:,:,np.newaxis], f)