import os
import time
import ntpath
import numpy as np
from PIL import Image
from os.path import join, exists
from keras.models import model_from_json
from utils.data_utils import getPaths, read_and_resize, preprocess, deprocess, get_local_test_data
import subprocess

test_paths = [r"C:\Users\vbvsi\Pictures\Screenshots\img.png"]
samples_dir = "data"
if not exists(samples_dir):
    os.makedirs(samples_dir)

model_h5 = r"C:\Users\vbvsi\Downloads\FUnieGan-main\FUnieGan-main\models\gen_p\model_15320_.h5"
model_json = r"C:\Users\vbvsi\Downloads\FUnieGan-main\FUnieGan-main\models\gen_p\model_15320_.json"

try:
    assert (exists(model_h5) and exists(model_json))

    with open(model_json, "r") as json_file:
        loaded_model_json = json_file.read()
    funie_gan_generator = model_from_json(loaded_model_json)
    funie_gan_generator.load_weights(model_h5)
    print("\nLoaded data and model")

    times = []
    s = time.time()
    for img_path in test_paths:
        inp_img = read_and_resize(img_path, (256, 256))
        im = preprocess(inp_img)
        im = np.expand_dims(im, axis=0)
        s = time.time()
        gen = funie_gan_generator.predict(im)
        gen_img = deprocess(gen)[0]
        tot = time.time() - s
        times.append(tot)
        img_name = ntpath.basename(img_path)
        out_img = np.hstack((inp_img, gen_img)).astype('uint8')
        Image.fromarray(out_img).save(join(samples_dir, img_name))

    num_test = len(test_paths)
    if num_test == 0:
        print("\nFound no images for test")
    else:
        Ttime, Mtime = np.sum(times[1:]), np.mean(times[1:])
        print("Time taken: {0} sec at {1} fps".format(Ttime, 1. / Mtime))
        print("\nSaved generated images in {0}\n".format(samples_dir))

except Exception as e:
    print(f"Model Loaded {e}. Executing inference.")
    subprocess.run(["python", "test.py"])