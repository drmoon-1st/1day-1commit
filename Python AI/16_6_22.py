# import sys

# def get_ans(val):
#     if val < 2:
#         return False
#     for i in range(2, int(val**0.5)+1):
#         if val % i == 0:
#             return False
#     return True


# p = int(sys.stdin.readline())
# while p:
#     a = int(sys.stdin.readline())
#     n1 = a // 2
#     n2 = a // 2
#     for i in range(a//2):
#         if get_ans(n1) and get_ans(n2):
#             print(n1, n2)
#             break
#         else:
#             n1 -= 1
#             n2 += 1
#     p -= 1

import os
import shutil

original_datasets = 'trainorigin'

base_dir = 'train'
# os.mkdir(base_dir)
train_dir = os.path.join(base_dir, 'train')
# os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
# os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
# os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir, 'cats')
# os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir, 'dogs')
# os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, 'cats')
# os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir, 'dogs')
# os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats')
# os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir, 'dogs')
# os.mkdir(test_dogs_dir)

# frames = ['cat.{}.jpg'.format(i) for i in range(1000)]
# for frame in frames:
#     src = os.path.join(original_datasets, frame)
#     dst = os.path.join(train_cats_dir, frame)
#     shutil.copyfile(src, dst)

# frames = ['cat.{}.jpg'.format(i) for i in range(1000,1500)]
# for frame in frames:
#     src = os.path.join(original_datasets, frame)
#     dst = os.path.join(validation_cats_dir, frame)
#     shutil.copyfile(src, dst)

# frames = ['cat.{}.jpg'.format(i) for i in range(1500,2000)]
# for frame in frames:
#     src = os.path.join(original_datasets, frame)
#     dst = os.path.join(test_cats_dir, frame)
#     shutil.copyfile(src, dst)

# frames = ['dog.{}.jpg'.format(i) for i in range(2000, 3000)]
# for frame in frames:
#     src = os.path.join(original_datasets, frame)
#     dst = os.path.join(train_dogs_dir, frame)
#     shutil.copyfile(src, dst)

# frames = ['dog.{}.jpg'.format(i) for i in range(3000,3500)]
# for frame in frames:
#     src = os.path.join(original_datasets, frame)
#     dst = os.path.join(validation_dogs_dir, frame)
#     shutil.copyfile(src, dst)

# frames = ['dog.{}.jpg'.format(i) for i in range(3500,4000)]
# for frame in frames:
#     src = os.path.join(original_datasets, frame)
#     dst = os.path.join(test_dogs_dir, frame)
#     shutil.copyfile(src, dst)

from tensorflow.keras import layers
from tensorflow.keras import models

# model = models.Sequential()
# model.add(layers.Conv2D(32, 3, activation='relu', input_shape=(150, 150, 3)))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64, 3, activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(128, 3, activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Flatten())
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))

from tensorflow.keras import optimizers
# model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=1e-4))

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# train_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=20, class_mode='binary')
# validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=20, class_mode='binary')

# history = model.fit(train_generator, steps_per_epoch=100, epochs=30, validation_data=validation_generator, validation_steps=50)
# model.save('cats_and_dogs_small_1.h5')

# dat_gen = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.1,
#     zoom_range=0.1,
#     horizontal_flip=True,
#     fill_mode='nearest')

from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# frames = sorted([os.path.join(train_cats_dir, frame) for frame in os.listdir(train_cats_dir)])

# img_path = frames[3]
# img = image.load_img(img_path, target_size=(150,150))

# x = image.img_to_array(img)
# x = x.reshape((1,)+x.shape)

# i = 0
# for batch in dat_gen.flow(x, batch_size=1):
#     plt.figure(i)
#     imgplot = plt.imshow(image.array_to_img(batch[0]))
#     i += 1
#     if i % 4 == 0:
#         break
# plt.show()

# model = models.Sequential()
# model.add(layers.Conv2D(32, 3, activation='relu', input_shape=(150, 150, 3)))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64, 3, activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(128, 3, activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(128, 3, activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Flatten())
# model.add(layers.Dropout((0.5)))
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))

# from tensorflow.keras import optimizers
# model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=1e-4))

# train_datgen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,)

# test_datagen = ImageDataGenerator(rescale=1./255)
# train_generator = train_datgen.flow_from_directory(train_dir, target_size=(150,150), batch_size=32, class_mode='binary')
# validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150,150), batch_size=32, class_mode='binary')

# history = model.fit_generator(train_generator, steps_per_epoch=50, epochs=50, validation_data=validation_generator, validation_steps=50)
# model.save('cats_and_dogs_small_1.h5')
# print(history)

from tensorflow.keras.applications import VGG16

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
# conv_base.summary()

import numpy as np

# datagen = ImageDataGenerator(rescale=1./255)
# batch_size = 20

# def extract_features(directory, sample_count):
#     features = np.zeros(shape=(sample_count, 4, 4, 512))
#     labels = np.zeros(shape=(sample_count))
#     generator = datagen.flow_from_directory(directory, target_size=(150, 150), batch_size=batch_size, class_mode='binary')
#     i = 0
#     for input_batch, label_batch in generator:
#         features_batch = conv_base.predict(input_batch)
#         features[i * batch_size : (i + 1) * batch_size] = features_batch
#         labels[i * batch_size : (i + 1) * batch_size] = label_batch
#         i += 1
#         if i * batch_size >= sample_count:
#             break
#     return features, labels

# train_features, train_labels = extract_features(train_dir, 2000)
# validation_features, validation_labels = extract_features(validation_dir, 1000)
# test_features, test_labels = extract_features(test_dir, 1000)

# train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
# validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
# test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

# model = models.Sequential()
# model.add(layers.Dense(256, activation='relu', input_dim=4*4*512))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(1, activation='sigmoid'))

# model.compile(optimizer=optimizers.RMSprop(learning_rate=2e-5), loss='binary_crossentropy', metrics=['accuracy'])
# history = model.fit(train_features, train_labels, epochs=30, batch_size=20, validation_data=(validation_features, validation_labels))

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(1, len(acc)+1)

# plt.plot(epochs, acc, 'bo')
# plt.plot(epochs, val_acc, 'b')

# plt.figure()

# plt.plot(epochs, loss, 'bo')
# plt.plot(epochs, val_loss, 'b')
# plt.show()

# model = models.Sequential()
# model.add(conv_base)
# model.add(layers.Flatten())
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))

# conv_base.trainable = False

# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     height_shift_range=0.1,
#     width_shift_range=0.1,
#     shear_range=0.1,
#     zoom_range=0.1,
#     horizontal_flip=True,
#     fill_mode='nearest')

# test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(150,150),
#     batch_size=20,
#     class_mode='binary')

# validation_generator = test_datagen.flow_from_directory(
#     validation_dir,
#     target_size=(150,150),
#     batch_size=20,
#     class_mode='binary')

# model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=2e-5), metrics=['acc'])

# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=100,
#     epochs=30,
#     validation_data=validation_generator,
#     validation_steps=50,
#     verbose=2)

# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(1, len(acc)+1)

# plt.plot(epochs, acc, 'bo')
# plt.plot(epochs, val_acc, 'b')

# plt.figure()

# plt.plot(epochs, loss, 'bo')
# plt.plot(epochs, val_loss, 'b')
# plt.show()

# model = models.Sequential()
# model.add(conv_base)
# model.add(layers.Flatten())
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))

# conv_base.trainable = True

# set_trainable = False

# for layer in conv_base.layers:
#     if layer.name == 'block5_conv1':
#         set_trainable = True
#     if set_trainable:
#         layer.trainable = True
#     else:
#         layer.trainable = False

# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     height_shift_range=0.1,
#     width_shift_range=0.1,
#     shear_range=0.1,
#     zoom_range=0.1,
#     horizontal_flip=True,
#     fill_mode='nearest')

# test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(150,150),
#     batch_size=20,
#     class_mode='binary')

# validation_generator = test_datagen.flow_from_directory(
#     validation_dir,
#     target_size=(150,150),
#     batch_size=20,
#     class_mode='binary')

# model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=2e-5), metrics=['accuracy'])

# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=100,
#     epochs=100,
#     validation_data=validation_generator,
#     validation_steps=50)

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(1, len(acc)+1)

# plt.plot(epochs, acc, 'bo')
# plt.plot(epochs, val_acc, 'b')

# plt.figure()

# plt.plot(epochs, loss, 'bo')
# plt.plot(epochs, val_loss, 'b')
# plt.show()

from keras import backend as k

model = VGG16(weights='imagenet', include_top=False)
layer_name = 'block3_conv1'
filter_index = 0
layer_output = model.get_layer(layer_name).output
loss = k.mean(layer_output[:, :, :, filter_index])

grad = k.gradients(loss, model.input[0])
grad /= (k.sqrt(k.mean(k.square(grad))) + 1e-5)