import os, zipfile
import tensorflow as tf


try:
    os.mkdir('data')
except:
    pass

folder = './data'
file = 'TrainPhotosRecodedv1.zip'

url = "https://github.com/FloodCamML/FloodCam-WUSFeats/releases/download/v0.0.0/"+file
filename = os.path.join(os.getcwd(), file)
print("Downloading %s ... " % (filename))
tf.keras.utils.get_file(filename, url)
print("Unzipping to %s ... " % (folder))
with zipfile.ZipFile(file, "r") as z_fp:
    z_fp.extractall("./"+folder)



try:
    os.remove(file)
except:
    pass