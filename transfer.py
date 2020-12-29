
'''built using tensorflow's style transfer tutorial'''

import os
import time

import PIL.Image

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import tensorflow as tf

from os.path import expanduser
home = expanduser("~")

content_layers = ['block5_conv2']
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

total_variation_weight=30

class StyleContentModel(tf.keras.models.Model):
    def __init__(self):
        super(StyleContentModel, self).__init__()
        self.vgg = vggLayers(style_layers+content_layers)
        self.vgg.trainable = False
        self.style_weight = 1e-2
        self.content_weight = 1e-2


    def call(self, inputs):
        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs = outputs[:len(style_layers)]
        content_outputs = outputs[len(style_layers):]

        style_outputs = [gramMatrix(style_output) for style_output in style_outputs]

        style_dict = {name: output for name, output in zip(style_layers, style_outputs)}
        content_dict = {name: output for name, output in zip(content_layers, content_outputs)}

        return {'content':content_dict, 'style':style_dict}

    def getTargets(self, content_image, style_image):
        self.style_targets = self.call(style_image)['style']
        self.content_targets = self.call(content_image)['content'] 

    def styleContentLoss(self, outputs):

        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-self.style_targets[name])**2) 
                               for name in style_outputs.keys()])
        style_loss *= self.style_weight / len(style_outputs)

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-self.content_targets[name])**2) 
                                 for name in content_outputs.keys()])
        content_loss *= self.content_weight / len(content_outputs)
        loss = style_loss + content_loss
        return loss

    @tf.function()
    def train_step(self,image):
        with tf.GradientTape() as tape:
            outputs = self.call(image)
            loss = self.styleContentLoss(outputs)
            loss += total_variation_weight*tf.image.total_variation(image)
            grad = tape.gradient(loss, image)
        self.opt.apply_gradients([(grad,image)])
        image.assign(clip0_1(image))

    def train(self,image, epochs, steps_per_epoch, opt):
        self.opt = opt
        start = time.time()

        step = 0
        for n in range(epochs):
            for m in range(steps_per_epoch):
                step += 1
                self.train_step(image)
                print(".", end='')
            print("Train step: {}".format(step))

        end = time.time()
        print("Total time: {:.1f}".format(end-start))


def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def tensorToImage(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def classifyWithVgg(image):
    x = tf.keras.applications.vgg19.preprocess_input(image)
    x = tf.image.resize(x, (224, 224))
    vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
    prediction_probabilities = vgg(x)

    predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
    [(class_name, prob) for (number, class_name, prob) in predicted_top_5]

    print(predicted_top_5)\


def vggLayers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    outputs = [vgg.get_layer(name).output for name in layer_names]
    vgg.trainable = False

    model = tf.keras.Model([vgg.input], outputs)

    return model

def printVggLayers(layers, outputs):
    for name, output in zip(layers, outputs):
        print(name)
        print("  shape: ", output.numpy().shape)
        print("  min: ", output.numpy().min())
        print("  max: ", output.numpy().max())
        print("  mean: ", output.numpy().mean())
        print()

def printModelOutputDetails(model, image):
    results = model.call(tf.constant(image))

    print('Styles:')
    for name, output in sorted(results['style'].items()):
        print("  ", name)
        print("    shape: ", output.numpy().shape)
        print("    min: ", output.numpy().min())
        print("    max: ", output.numpy().max())
        print("    mean: ", output.numpy().mean())
        print()

    print("Contents:")
    for name, output in sorted(results['content'].items()):
        print("  ", name)
        print("    shape: ", output.numpy().shape)
        print("    min: ", output.numpy().min())
        print("    max: ", output.numpy().max())
        print("    mean: ", output.numpy().mean()) 

def gramMatrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)

    return result/num_locations

def clip0_1(image):
     return tf.clip_by_value(image, clip_value_min=0, clip_value_max=1)


def transfer(mainImageUrl, styleImageUrl):

    mpl.rcParams['figure.figsize'] = (12,12)
    mpl.rcParams['axes.grid'] = False

    opt = tf.optimizers.Adam(learning_rate=0.006, beta_1=0.99, epsilon=1e-1)
    try:
        os.remove(home+'/.keras/datasets/NeuralStyleTransferMain')
        os.remove(home+'/.keras/datasets/NeuralStyleTransferStyle')
    except:
        pass

    content_path = tf.keras.utils.get_file('NeuralStyleTransferMain', mainImageUrl)
    style_path = tf.keras.utils.get_file('NeuralStyleTransferStyle',styleImageUrl)

    content_image = load_img(content_path)
    style_image = load_img(style_path)

    # classifyWithVgg(content_image)

    model = StyleContentModel()
    model.getTargets(content_image, style_image)
    image = tf.Variable(content_image)

    # printVggLayers(style_layers, style_output)
    # printModelOutputDetails(model, image)

    model.train(image, 10, 300, opt)


    plt.subplot(2, 2, 1)
    imshow(content_image, 'Content Image')
    plt.subplot(2, 2, 2)
    imshow(style_image, 'Style Image')
    plt.subplot(2, 2, 3)
    imshow(np.asarray(image), 'result')
    plt.show()

    return tensorToImage(image)

