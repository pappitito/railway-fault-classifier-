from locale import MON_10
from tokenize import String
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.preprocessing import image
import numpy as np

def readclass(thearray, counter):
    for item in thearray:
        if item[0] == max(item):
            print('the railway in image ', counter, ' is defective')
        if item[1] == max(item):
            print('the railway in image ', counter, ' is Non defective')

class mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (logs.get('accuracy') >= 0.78):
           self.model.stop_training = True

train_dir = '/Users/mac/Documents/work life/programming/machine learning/models/faulty railway/railway-fault/train'
valid_dir = '/Users/mac/Documents/work life/programming/machine learning/models/faulty railway/railway-fault/valid'
imag_paths = [
          '/Users/mac/Documents/work life/programming/machine learning/models/faulty railway/railway-fault/test/Defective/IMG_20201114_100209.jpg',
          '/Users/mac/Documents/work life/programming/machine learning/models/faulty railway/railway-fault/test/Defective/IMG_20201114_101124.jpg',
        '/Users/mac/Documents/work life/programming/machine learning/models/faulty railway/railway-fault/test/Defective/IMG_20201114_101200.jpg',
        '/Users/mac/Documents/work life/programming/machine learning/models/faulty railway/railway-fault/test/Defective/IMG_20201114_103110.jpg',
        '/Users/mac/Documents/work life/programming/machine learning/models/faulty railway/railway-fault/test/Defective/IMG_20201114_102222.jpg',
        '/Users/mac/Documents/work life/programming/machine learning/models/faulty railway/railway-fault/test/Defective/IMG_20201211_121712_1.jpg',
         '/Users/mac/Documents/work life/programming/machine learning/models/faulty railway/railway-fault/test/Defective/IMG_20201114_102203.jpg',
          '/Users/mac/Documents/work life/programming/machine learning/models/faulty railway/railway-fault/test/Defective/IMG_20201114_102222.jpg',
          '/Users/mac/Documents/work life/programming/machine learning/models/faulty railway/railway-fault/test/Non defective/IMG_20201114_101756.jpg',
        '/Users/mac/Documents/work life/programming/machine learning/models/faulty railway/railway-fault/test/Non defective/IMG_20201114_101907.jpg',
        '/Users/mac/Documents/work life/programming/machine learning/models/faulty railway/railway-fault/test/Non defective/IMG_20201114_102431.jpg',   
        '/Users/mac/Documents/work life/programming/machine learning/models/faulty railway/railway-fault/test/Non defective/IMG_20201114_100358.jpg',
        '/Users/mac/Documents/work life/programming/machine learning/models/faulty railway/railway-fault/test/Non defective/IMG_20201114_100344.jpg', 
        '/Users/mac/Documents/work life/programming/machine learning/models/faulty railway/railway-fault/test/Non defective/IMG_20201114_102945.jpg',
        '/Users/mac/Documents/work life/programming/machine learning/models/faulty railway/railway-fault/test/Non defective/IMG_20201114_102909.jpg',
        '/Users/mac/Documents/work life/programming/machine learning/models/faulty railway/railway-fault/test/Non defective/IMG_20201114_100023.jpg' 
          ]

train_gen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    fill_mode = 'nearest',
    rotation_range = 40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.4
)

valid_gen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    fill_mode = 'nearest',
    rotation_range = 40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.4
)

train_data = train_gen.flow_from_directory(train_dir, target_size = (224,224), class_mode = 'binary')

valid_data = valid_gen.flow_from_directory(valid_dir, target_size = (224,224), class_mode = 'binary')

model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(16, (3,3), activation='relu',
                  input_shape=(224, 224, 3)),
      tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512, activation='relu'),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(32, activation='relu'),
      tf.keras.layers.Dense(2, activation='softmax')
])

callbacking = mycallback()
model.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(learning_rate=0.001), metrics=['accuracy'])

train_model = model.fit(train_data, epochs = 70, callbacks= callbacking, validation_data= valid_data )

counter = 0

for images in imag_paths:
    path = images
    img = tf.keras.utils.load_img(path, target_size= (224,224))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images_test = np.vstack([x])
    classes = model.predict(images_test)
    print(classes)
    counter += 1
    readclass(classes,counter)
    




