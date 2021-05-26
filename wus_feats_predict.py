


#i/o
import requests, os, random
from glob import glob
from collections import Counter
from collections import defaultdict
from PIL import Image
from skimage.io import imread
import pickle

#numerica
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers
from skimage.transform import resize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA  #for data dimensionality reduction / viz.


# plots
# from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns #extended functionality / style to matplotlib plots
from matplotlib.offsetbox import OffsetImage, AnnotationBbox #for visualizing image thumbnails plotted as markers