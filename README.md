# frame interpolation attempt using CNN autoencoder

## model summary
<pre>
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 128, 128, 4)       76        
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 64, 64, 4)         0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 64, 64, 8)         296       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 32, 32, 8)         0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 32, 32, 16)        1168      
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 16, 16, 16)        0         
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr (None, 32, 32, 16)        2320      
_________________________________________________________________
conv2d_transpose_2 (Conv2DTr (None, 64, 64, 4)         580       
_________________________________________________________________
conv2d_transpose_3 (Conv2DTr (None, 128, 128, 1)       37        
=================================================================
Total params: 4,477
Trainable params: 4,477
Non-trainable params: 0
_________________________________________________________________
</pre>

## training result
Loss and validation loss.     
training summary (lr=1E-6 for 600 epochs batch_size=32)   
![train result](train.png?raw=true)   
additional training summary (lr=1E-8 for 40000 epochs batch_size=64)   
![train result 1](train1.png?raw=true)   

## evaluation result
Prediction of the in-between frame.    
Seems that the model overfitted and creates a image similar to the first image.  
    
evaluation result   
![evaluation result](test.png?raw=true)   
failed prediction   
![evaluation fail](fail.png?raw=true)   