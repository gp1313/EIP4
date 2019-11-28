Accuracy on test data is: 82.38
# Define the model
model = Sequential()

model.add(SeparableConv2D(48, 3, 3, border_mode='same', input_shape=(32, 32, 3))) # 32 3
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(SeparableConv2D(48, 3, 3)) # 30 5
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2))) # 15 6
model.add(Dropout(0.2))

model.add(SeparableConv2D(96, 3, 3, border_mode='same')) # 15 10
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(SeparableConv2D(96, 3, 3)) # 13 14
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2))) # 6 16
model.add(Dropout(0.2))

model.add(SeparableConv2D(192, 3, 3, border_mode='same')) # 6 24
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(SeparableConv2D(192, 3, 3)) # 4 32
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2))) # 2 36
model.add(Dropout(0.2))

model.add(SeparableConv2D(num_classes, 1, 1)) # 2 36
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(GlobalAveragePooling2D()) # 1 36

model.add(Activation('softmax'))
Epoch 1/50
390/390 [==============================] - 31s 78ms/step - loss: 1.5087 - acc: 0.4802 - val_loss: 1.2457 - val_acc: 0.5673
Epoch 2/50
390/390 [==============================] - 27s 68ms/step - loss: 1.1365 - acc: 0.6201 - val_loss: 1.0927 - val_acc: 0.6263
Epoch 3/50
390/390 [==============================] - 27s 69ms/step - loss: 0.9703 - acc: 0.6771 - val_loss: 0.9167 - val_acc: 0.6901
Epoch 4/50
390/390 [==============================] - 27s 69ms/step - loss: 0.8659 - acc: 0.7107 - val_loss: 0.9090 - val_acc: 0.6986
Epoch 5/50
390/390 [==============================] - 27s 69ms/step - loss: 0.7893 - acc: 0.7371 - val_loss: 0.8181 - val_acc: 0.7194
Epoch 6/50
390/390 [==============================] - 27s 68ms/step - loss: 0.7345 - acc: 0.7542 - val_loss: 0.7613 - val_acc: 0.7469
Epoch 7/50
390/390 [==============================] - 27s 68ms/step - loss: 0.6944 - acc: 0.7663 - val_loss: 0.7133 - val_acc: 0.7583
Epoch 8/50
390/390 [==============================] - 27s 68ms/step - loss: 0.6581 - acc: 0.7785 - val_loss: 0.8109 - val_acc: 0.7262
Epoch 9/50
390/390 [==============================] - 27s 68ms/step - loss: 0.6215 - acc: 0.7902 - val_loss: 0.6420 - val_acc: 0.7854
Epoch 10/50
390/390 [==============================] - 27s 68ms/step - loss: 0.6032 - acc: 0.7949 - val_loss: 0.7194 - val_acc: 0.7564
Epoch 11/50
390/390 [==============================] - 27s 69ms/step - loss: 0.5746 - acc: 0.8054 - val_loss: 0.6971 - val_acc: 0.7665
Epoch 12/50
390/390 [==============================] - 27s 69ms/step - loss: 0.5583 - acc: 0.8094 - val_loss: 0.6367 - val_acc: 0.7860
Epoch 13/50
390/390 [==============================] - 27s 69ms/step - loss: 0.5380 - acc: 0.8169 - val_loss: 0.6152 - val_acc: 0.7902
Epoch 14/50
390/390 [==============================] - 27s 69ms/step - loss: 0.5226 - acc: 0.8211 - val_loss: 0.5840 - val_acc: 0.8039
Epoch 15/50
390/390 [==============================] - 27s 69ms/step - loss: 0.5019 - acc: 0.8280 - val_loss: 0.6725 - val_acc: 0.7735
Epoch 16/50
390/390 [==============================] - 27s 68ms/step - loss: 0.4955 - acc: 0.8305 - val_loss: 0.6171 - val_acc: 0.7941
Epoch 17/50
390/390 [==============================] - 27s 69ms/step - loss: 0.4820 - acc: 0.8340 - val_loss: 0.5710 - val_acc: 0.8053
Epoch 18/50
390/390 [==============================] - 27s 69ms/step - loss: 0.4673 - acc: 0.8395 - val_loss: 0.6789 - val_acc: 0.7771
Epoch 19/50
390/390 [==============================] - 27s 68ms/step - loss: 0.4593 - acc: 0.8415 - val_loss: 0.5749 - val_acc: 0.8078
Epoch 20/50
390/390 [==============================] - 27s 69ms/step - loss: 0.4471 - acc: 0.8458 - val_loss: 0.6009 - val_acc: 0.8045
Epoch 21/50
390/390 [==============================] - 27s 68ms/step - loss: 0.4367 - acc: 0.8488 - val_loss: 0.5555 - val_acc: 0.8160
Epoch 22/50
390/390 [==============================] - 27s 68ms/step - loss: 0.4275 - acc: 0.8537 - val_loss: 0.5517 - val_acc: 0.8113
Epoch 23/50
390/390 [==============================] - 26s 68ms/step - loss: 0.4217 - acc: 0.8559 - val_loss: 0.6814 - val_acc: 0.7773
Epoch 24/50
390/390 [==============================] - 27s 68ms/step - loss: 0.4135 - acc: 0.8579 - val_loss: 0.5441 - val_acc: 0.8172
Epoch 25/50
390/390 [==============================] - 27s 69ms/step - loss: 0.4013 - acc: 0.8617 - val_loss: 0.5730 - val_acc: 0.8120
Epoch 26/50
390/390 [==============================] - 27s 68ms/step - loss: 0.3967 - acc: 0.8627 - val_loss: 0.5757 - val_acc: 0.8076
Epoch 27/50
390/390 [==============================] - 27s 68ms/step - loss: 0.3885 - acc: 0.8668 - val_loss: 0.5514 - val_acc: 0.8164
Epoch 28/50
390/390 [==============================] - 27s 69ms/step - loss: 0.3850 - acc: 0.8668 - val_loss: 0.6192 - val_acc: 0.7984
Epoch 29/50
390/390 [==============================] - 27s 68ms/step - loss: 0.3763 - acc: 0.8696 - val_loss: 0.5516 - val_acc: 0.8201
Epoch 30/50
390/390 [==============================] - 27s 68ms/step - loss: 0.3712 - acc: 0.8715 - val_loss: 0.5604 - val_acc: 0.8213
Epoch 31/50
390/390 [==============================] - 27s 68ms/step - loss: 0.3710 - acc: 0.8703 - val_loss: 0.5534 - val_acc: 0.8208
Epoch 32/50
390/390 [==============================] - 27s 68ms/step - loss: 0.3637 - acc: 0.8742 - val_loss: 0.5739 - val_acc: 0.8145
Epoch 33/50
390/390 [==============================] - 27s 69ms/step - loss: 0.3549 - acc: 0.8778 - val_loss: 0.5446 - val_acc: 0.8205
Epoch 34/50
390/390 [==============================] - 27s 68ms/step - loss: 0.3490 - acc: 0.8777 - val_loss: 0.5523 - val_acc: 0.8238
Epoch 35/50
390/390 [==============================] - 27s 68ms/step - loss: 0.3466 - acc: 0.8796 - val_loss: 0.5334 - val_acc: 0.8285
Epoch 36/50
390/390 [==============================] - 27s 68ms/step - loss: 0.3428 - acc: 0.8815 - val_loss: 0.5664 - val_acc: 0.8192
Epoch 37/50
390/390 [==============================] - 27s 69ms/step - loss: 0.3350 - acc: 0.8837 - val_loss: 0.5748 - val_acc: 0.8169
Epoch 38/50
390/390 [==============================] - 27s 68ms/step - loss: 0.3356 - acc: 0.8830 - val_loss: 0.5520 - val_acc: 0.8230
Epoch 39/50
390/390 [==============================] - 26s 68ms/step - loss: 0.3299 - acc: 0.8853 - val_loss: 0.5366 - val_acc: 0.8276
Epoch 40/50
390/390 [==============================] - 27s 68ms/step - loss: 0.3244 - acc: 0.8874 - val_loss: 0.5436 - val_acc: 0.8249
Epoch 41/50
390/390 [==============================] - 27s 68ms/step - loss: 0.3226 - acc: 0.8884 - val_loss: 0.6242 - val_acc: 0.8070
Epoch 42/50
390/390 [==============================] - 27s 69ms/step - loss: 0.3166 - acc: 0.8888 - val_loss: 0.5565 - val_acc: 0.8238
Epoch 43/50
390/390 [==============================] - 27s 68ms/step - loss: 0.3144 - acc: 0.8900 - val_loss: 0.5330 - val_acc: 0.8274
Epoch 44/50
390/390 [==============================] - 27s 68ms/step - loss: 0.3139 - acc: 0.8898 - val_loss: 0.5554 - val_acc: 0.8204
Epoch 45/50
390/390 [==============================] - 27s 68ms/step - loss: 0.3101 - acc: 0.8916 - val_loss: 0.5320 - val_acc: 0.8315
Epoch 46/50
390/390 [==============================] - 27s 68ms/step - loss: 0.3024 - acc: 0.8933 - val_loss: 0.5459 - val_acc: 0.8261
Epoch 47/50
390/390 [==============================] - 27s 69ms/step - loss: 0.3030 - acc: 0.8938 - val_loss: 0.5780 - val_acc: 0.8184
Epoch 48/50
390/390 [==============================] - 27s 68ms/step - loss: 0.2995 - acc: 0.8959 - val_loss: 0.5493 - val_acc: 0.8223
Epoch 49/50
390/390 [==============================] - 27s 68ms/step - loss: 0.2979 - acc: 0.8976 - val_loss: 0.5522 - val_acc: 0.8287
Epoch 50/50
390/390 [==============================] - 27s 68ms/step - loss: 0.2938 - acc: 0.8983 - val_loss: 0.5677 - val_acc: 0.8238
Model took 1340.15 seconds to train
