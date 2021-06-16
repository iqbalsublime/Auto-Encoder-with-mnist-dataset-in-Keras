

```python
import keras
from keras import layers
```


```python
encoding_dim = 32 
```


```python
input_img = keras.Input(shape=(784,))
```


```python
encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
```


```python
decoded = layers.Dense(784, activation='sigmoid')(encoded)
```


```python
autoencoder = keras.Model(input_img, decoded)
```


```python
encoder = keras.Model(input_img, encoded)
```


```python
encoded_input = keras.Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
```


```python
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
```


```python
from keras.datasets import mnist
import numpy as np 
(x_train, _),(x_test, _) = mnist.load_data()
```


```python
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape(len(x_train), np.prod(x_train.shape[1:]))
x_test = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))
print(x_train.shape)
print(x_test.shape)
```

    (60000, 784)
    (10000, 784)
    


```python
autoencoder.fit(x_train, x_train, 
               epochs=50, 
               batch_size=256, 
               shuffle=True,
               validation_data=(x_test, x_test))
```

    Train on 60000 samples, validate on 10000 samples
    Epoch 1/50
    60000/60000 [==============================] - 3s 45us/step - loss: 0.2733 - val_loss: 0.1884
    Epoch 2/50
    60000/60000 [==============================] - 2s 39us/step - loss: 0.1706 - val_loss: 0.1538
    Epoch 3/50
    60000/60000 [==============================] - 2s 39us/step - loss: 0.1443 - val_loss: 0.1338
    Epoch 4/50
    60000/60000 [==============================] - 2s 37us/step - loss: 0.1289 - val_loss: 0.1217
    Epoch 5/50
    60000/60000 [==============================] - 2s 37us/step - loss: 0.1187 - val_loss: 0.1131
    Epoch 6/50
    60000/60000 [==============================] - 3s 42us/step - loss: 0.1114 - val_loss: 0.1071
    Epoch 7/50
    60000/60000 [==============================] - 2s 38us/step - loss: 0.1063 - val_loss: 0.1028
    Epoch 8/50
    60000/60000 [==============================] - 2s 38us/step - loss: 0.1026 - val_loss: 0.0998
    Epoch 9/50
    60000/60000 [==============================] - 2s 37us/step - loss: 0.0999 - val_loss: 0.0975
    Epoch 10/50
    60000/60000 [==============================] - 2s 38us/step - loss: 0.0979 - val_loss: 0.0959
    Epoch 11/50
    60000/60000 [==============================] - 2s 38us/step - loss: 0.0966 - val_loss: 0.0947
    Epoch 12/50
    60000/60000 [==============================] - 2s 38us/step - loss: 0.0957 - val_loss: 0.0940
    Epoch 13/50
    60000/60000 [==============================] - 3s 42us/step - loss: 0.0951 - val_loss: 0.0936
    Epoch 14/50
    60000/60000 [==============================] - 2s 38us/step - loss: 0.0946 - val_loss: 0.0932
    Epoch 15/50
    60000/60000 [==============================] - 2s 37us/step - loss: 0.0943 - val_loss: 0.0929
    Epoch 16/50
    60000/60000 [==============================] - 2s 37us/step - loss: 0.0941 - val_loss: 0.0926
    Epoch 17/50
    60000/60000 [==============================] - 2s 38us/step - loss: 0.0939 - val_loss: 0.0925
    Epoch 18/50
    60000/60000 [==============================] - 2s 42us/step - loss: 0.0937 - val_loss: 0.0924
    Epoch 19/50
    60000/60000 [==============================] - 3s 43us/step - loss: 0.0936 - val_loss: 0.0924
    Epoch 20/50
    60000/60000 [==============================] - 2s 40us/step - loss: 0.0935 - val_loss: 0.0922
    Epoch 21/50
    60000/60000 [==============================] - 2s 41us/step - loss: 0.0934 - val_loss: 0.0922
    Epoch 22/50
    60000/60000 [==============================] - 2s 37us/step - loss: 0.0933 - val_loss: 0.0921
    Epoch 23/50
    60000/60000 [==============================] - 2s 41us/step - loss: 0.0933 - val_loss: 0.0921
    Epoch 24/50
    60000/60000 [==============================] - 2s 41us/step - loss: 0.0932 - val_loss: 0.0920
    Epoch 25/50
    60000/60000 [==============================] - 3s 42us/step - loss: 0.0932 - val_loss: 0.0919
    Epoch 26/50
    60000/60000 [==============================] - 3s 42us/step - loss: 0.0931 - val_loss: 0.0919
    Epoch 27/50
    60000/60000 [==============================] - 2s 41us/step - loss: 0.0931 - val_loss: 0.0919
    Epoch 28/50
    60000/60000 [==============================] - 2s 39us/step - loss: 0.0931 - val_loss: 0.0920
    Epoch 29/50
    60000/60000 [==============================] - 3s 43us/step - loss: 0.0930 - val_loss: 0.0919
    Epoch 30/50
    60000/60000 [==============================] - 3s 43us/step - loss: 0.0930 - val_loss: 0.0918
    Epoch 31/50
    60000/60000 [==============================] - 3s 43us/step - loss: 0.0930 - val_loss: 0.0918
    Epoch 32/50
    60000/60000 [==============================] - 2s 41us/step - loss: 0.0929 - val_loss: 0.0918
    Epoch 33/50
    60000/60000 [==============================] - 3s 43us/step - loss: 0.0929 - val_loss: 0.0918
    Epoch 34/50
    60000/60000 [==============================] - 2s 40us/step - loss: 0.0929 - val_loss: 0.0919
    Epoch 35/50
    60000/60000 [==============================] - 2s 41us/step - loss: 0.0929 - val_loss: 0.0917
    Epoch 36/50
    60000/60000 [==============================] - 2s 40us/step - loss: 0.0929 - val_loss: 0.0917
    Epoch 37/50
    60000/60000 [==============================] - 2s 40us/step - loss: 0.0929 - val_loss: 0.0917
    Epoch 38/50
    60000/60000 [==============================] - 2s 39us/step - loss: 0.0928 - val_loss: 0.0918
    Epoch 39/50
    60000/60000 [==============================] - 3s 43us/step - loss: 0.0928 - val_loss: 0.0917
    Epoch 40/50
    60000/60000 [==============================] - 3s 42us/step - loss: 0.0928 - val_loss: 0.0916
    Epoch 41/50
    60000/60000 [==============================] - 2s 40us/step - loss: 0.0928 - val_loss: 0.0917
    Epoch 42/50
    60000/60000 [==============================] - 2s 41us/step - loss: 0.0928 - val_loss: 0.0917
    Epoch 43/50
    60000/60000 [==============================] - 2s 40us/step - loss: 0.0928 - val_loss: 0.0917
    Epoch 44/50
    60000/60000 [==============================] - 2s 39us/step - loss: 0.0928 - val_loss: 0.0917
    Epoch 45/50
    60000/60000 [==============================] - 2s 40us/step - loss: 0.0927 - val_loss: 0.0916
    Epoch 46/50
    60000/60000 [==============================] - 2s 42us/step - loss: 0.0927 - val_loss: 0.0917
    Epoch 47/50
    60000/60000 [==============================] - 2s 40us/step - loss: 0.0927 - val_loss: 0.0916
    Epoch 48/50
    60000/60000 [==============================] - 2s 39us/step - loss: 0.0927 - val_loss: 0.0916
    Epoch 49/50
    60000/60000 [==============================] - 2s 40us/step - loss: 0.0927 - val_loss: 0.0916
    Epoch 50/50
    60000/60000 [==============================] - 2s 39us/step - loss: 0.0927 - val_loss: 0.0915
    




    <keras.callbacks.History at 0x24b21e209e8>




```python
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)
```


```python
import matplotlib.pyplot as plt
```


```python
n = 10 
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```


![png](output_14_0.png)



```python

```
