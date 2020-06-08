import tensorflow as tf
import tensorflow_hub as hub
 
from PIL import Image
import numpy as np
 
import json
 
import argparse
 
 
parser = argparse.ArgumentParser()
parser.add_argument("--image", help='./test_images, the file is in this folder',type=str)
parser.add_argument("--model", help= 'saved model',type=str)
parser.add_argument("--top_k", help='top K indices', default=1, type=int)
parser.add_argument("--category_name", help='JSON file',default='./label_map.json',type=str)
args = parser.parse_args()
 
image_path = args.image
model = tf.keras.models.load_model(args.model,custom_objects={'KerasLayer':hub.KerasLayer})
top_K = args.top_k
category_name = args.category_name
 
 
image_size = 224
 
with open(category_name, 'r') as f:
    class_names = json.load(f)
 
def process_image(image):
  image = tf.convert_to_tensor(image)
  image = tf.image.resize(image, (image_size, image_size))
  image /= 255
  image = image.numpy()
  return image
 
def predict(image_path,model,top_k):
  img = Image.open(image_path)
  image = np.asarray(img)
  image = process_image(image)
  image = np.expand_dims(image,axis=0)
  ps=model.predict(image)
  indices,values = tf.nn.top_k(ps,k=top_k)
  indices = indices.numpy()[0]
  values = values.numpy()[0]
  top_values = [class_names[str(value+1)] for value in values]
  return indices, top_values
 
probs, classes = predict(image_path,model,top_K)
print("Top Probability:", probs)
print("Top Class name: ", classes)
