from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
import time
import csv
import math

# https://keras.io/applications/#extract-features-with-vgg16

model = VGG16(weights='imagenet', include_top=False)

model.summary()
vgg16_features = []
start = time.time()
files = []
writers = []
for j in range(100):
    f = open('celeba/img_align_celeba_features_'+str(j).zfill(2)+'.dat', 'w')
    files.append(f)
    writers.append(csv.writer(f))
jo = 0
for i in range(1, 202600):
    j = math.floor(i / 2026)
    if jo != j:
        files[jo].close()
        print("Done writing to file number " + str(jo))
    jo = j
    writer = writers[j]
    img_path = 'celeba/images/img_align_celeba/'+str(i).zfill(6)+'.jpg'
    img = image.load_img(img_path, target_size=(178, 218))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    curr_feat = model.predict(img_data).flatten()
    vgg16_features.append(curr_feat)
    writer.writerow(curr_feat)
files[jo].close()
print("Done writing to file number " + str(jo))

#  for file in img_align_celeba_features_{00..99}.dat; do cat $file; done > img_align_celeba_features.dat
