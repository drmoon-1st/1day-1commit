from tensorflow.keras.applications import VGG16
from tensorflow.keras import backend as K

model = VGG16(weights='imagenet', include_top=False)

layer_name = 'block3_conv1'
filter_index = 0

layer_output = model.get_layer(layer_name).output
loss = K.mean(layer_output[:,:,:,filter_index])

grad = K.gradients(loss, model.input)[0]

grad /= (K.sqrt(K.mean(K.square(grad)))+1e-5)

iterate = K.function([model.input], [loss, grad])

import numpy as np
loss_vlaue, grad_value = iterate([np.zeros((1,150,150,3))])

input_img_data = np.random.random((1,150,150,3))*20+128.
step = 1.
for i in range(40):
    loss_value, grad_value = iterate([input_img_data])
    input_img_data = grad_value*step

def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    x += 0.5
    x - np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('utf8')
    return x

def generate_patterns(layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:,:,:,filter_index])

    grad = K.gradients(loss, model.input)[0]

    grad /= (K.sqrt(K.mean(K.square(grad)))+1e-5)

    iterate = K.function([model.input], [loss, grad])

    input_img_data = np.random.random((1,size,size,3))*20+128.

    step = 1.
    for i in range(40):
        loss_value, grad_value = iterate([input_img_data])
        input_img_data = grad_value*step
    
    img = input_img_data[0]
    return deprocess_image(img)

import matplotlib.pyplot as plt

plt.imshow(generate_patterns(layer_name, 0))
plt.show()