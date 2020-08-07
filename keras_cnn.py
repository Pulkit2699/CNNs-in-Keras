import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import copy
import operator
import pydot

def get_dataset(training=True):
  fashion_mnist = keras.datasets.fashion_mnist
  (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
  test_images = np.expand_dims(test_images, axis=3)
  train_images = np.expand_dims(train_images, axis=3)
  if training:
    return (train_images, train_labels)
  else:
    return (test_images, test_labels)
    
def build_model():
  model = keras.models.Sequential() 
  model.add(keras.layers.Conv2D(kernel_size=3, input_shape=(28,28,1), filters=64,activation = tf.nn.relu))
  model.add(keras.layers.Conv2D(kernel_size=3, filters=32,activation = tf.nn.relu))
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(10, activation = tf.nn.softmax))
  model.compile(loss='categorical_crossentropy',
                  optimizer = 'adam', 
                  metrics = ['accuracy'])
  return model
  
def train_model(model, train_img, train_lab, test_img, test_lab, T):
  train_lab = keras.utils.to_categorical(train_lab)
  test_lab = keras.utils.to_categorical(test_lab)
  model.fit(train_img, train_lab,validation_data=(test_img, test_lab),epochs = T)
  
  

def predict_label(model, images, index):
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  out = (model.predict(images))
  im = out[index]
  dict1 = {}
  for i in range(len(class_names)):
    dict1.update({class_names[i] : im[i]})
  sor = sorted(dict1.items(), key=operator.itemgetter(1))
  print(sor[len(class_names) - 1][0] + ": " + "{:.2f}".format(sor[len(class_names) - 1][1] * 100) + '%')
  print(sor[len(class_names) - 2][0] + ": " + "{:.2f}".format(sor[len(class_names) - 2][1] * 100) + '%')
  print(sor[len(class_names) - 3][0] + ": " + "{:.2f}".format(sor[len(class_names) - 3][1] * 100) + '%')

"""
(train_images, train_labels) = get_dataset()
(test_images, test_labels) = get_dataset(False)
model = build_model()
train_model(model, train_images, train_labels, test_images, test_labels, 5) 
predict_label(model, test_images, 2)
"""

"""
(train_images, train_labels) = get_dataset()
(test_images, test_labels) = get_dataset(False)
model = build_model()
print(train_model(model, train_images, train_labels, test_images, test_labels, 5))
"""
"""
model = build_model()
print(keras.utils.plot_model(model, to_file='model.png'))
"""
"""  
(test_images, test_labels) = get_dataset(False)
print(test_images.shape)
"""

  
def main():
  return
    
if __name__ == "__main__":
  main()
