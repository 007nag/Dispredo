import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.utils import plot_model
base_mobilenet_model = MobileNet(input_shape = (100,100,1), 
 include_top = False, weights = None)
mod= load_model("models/1/1.h5")
print(mod.summary())
print(base_mobilenet_model.summary())
"""plot_model(mod,to_file='modell.png',show_shapes=True, show_layer_names=True)
plot_model(base_mobilenet_model,to_file='mobilenet.png',show_shapes=True, show_layer_names=True)"""