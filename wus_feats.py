#!/usr/bin/env python
# coding: utf-8

## PLS DO NOT EDIT YET - NOT FULLY IMPLEMENTED 

# ## A Weakly Supervised Machine Learning Model for Flooded Road Recognition
# ### Dan Buscombe, Marda Science / USGS
# #### May, 2021; contribution to the COMET "Sunny Day Flooding" project

# The gist:
#
# * various image dims
# * various embedding dimensions
# * The model has 379,040 trainable parameters (no matter input image size)
#
# 1. Train a "weakly supervised" image embedding model
#   * the model is an autoencoder that creates an embedding feature for each image, such that that feature is maximally distant from all other features extracted from other classes
# 2. Evaluate the feature extractor
#   * study the model training history - the loss and accuracy curves of train and validation sets
# 3. Construct a "weakly supervised" classification model
#   * build a k-nearest-neighbour (kNN) classifier that classifies unseen imagery based on the k nearest neighbours to the current image.
#     (Or, more correctly, the nearest neighbour's of the image's embedding vector in the training set of embedding vectors)
# 4. Evaluate the classifier
#   * evaluate the performance of the trained model on the validation set
#   * plot a 'confusion matrix' of correspondences between actual and estimate class labels

#compare against a completely unsupervised approach, PCA, followed by kNN

# Number of interesting properties of an embedding model like this:
# * Number of model parameters doesn't increase with input image size
# * hyperparameters  of embedding model: num_embedding_dims, image size, 'temperature' of logit scaling
# * hyperparameters  of embedding model: number of batches, epochs, lr
# * hyperparameters  of kNN model: number of nearest neighbours
# * l2_normalize or not? just training or testing set too?
# * weakly supervised feature extraction followed by weakly supervised classification
# * amenable to k-NN which is a very intuitive and fast inference technique
# * selection of multiple k for kNN lends itself to simple ensemble predictions
# * use a constant learning rate (a scheduler doesn't result in better results; this model is more stable with a constant learning rate, which becomes an important tunable hyperparameter)
#
# Interesting having a 'not sure' category - asking the model to replicate human uncertainty

# Common deep neural networks used for image recognition employ an extremely discriminative approach that explicitly maps the classes to the image features, and optimized to extract the features that explicitly predict the class.
#
# Here, we will use a deep neural network to extract features based on those that maximize the distance between classes in feature space. This isn't the same level of 'supervision' in network training - instead of extracting features that predict the class, features are extracted so they are maximally similar to features from other images in the same class, and maximally distant from features in all other classes. There is no mapping from features to class. Only feature extraction based on a knowledge of which images are in the same class. Therefore this approach is known as 'weakly supervised' image feature extraction. The network we use is an example of an 'autoencoder' that embeds the information in the image into a lower dimensional space. Therefore the extracted features are called 'embeddings'.
#
# Nor does this feature extraction result in classification directly - we don't use a classifying head to inform how image features are extracted. So, we have to utilize another model to carry out classification. We use perhaps the simplest, conceptually; K-nearest neighbours.
#The idea is that it will cluster those embeddings (extracted features) and classification is based on the class of the K nearest neighbours with the K most similar embeddings.


#TODO: make a conda or requirements.txt


#i/o
import requests, os, random
from glob import glob
from collections import Counter
from collections import defaultdict
from PIL import Image
from skimage.io import imread

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


##==============================================

def standardize(img):
    #standardization using adjusted standard deviation
    N = np.shape(img)[0] * np.shape(img)[1]
    s = np.maximum(np.std(img), 1.0/np.sqrt(N))
    m = np.mean(img)
    img = (img - m) / s
    img = rescale(img, 0, 1)
    del m, s, N

    if np.ndim(img)!=3:
        img = np.dstack((img,img,img))

    return img

def rescale(dat,
    mn,
    mx):
    '''
    rescales an input dat between mn and mx
    '''
    m = min(dat.flatten())
    M = max(dat.flatten())
    return (mx-mn)*(dat-m)/(M-m)+mn


#-----------------------------------
def plot_one_class(inp_batch, sample_idx, label, batch_size, CLASSES, rows=8, cols=8, size=(20,15)):
    """
    plot_one_class(inp_batch, sample_idx, label, batch_size, CLASSES, rows=8, cols=8, size=(20,15)):
    Plot "batch_size" images that belong to the class "label"
    INPUTS:
        * inp_batch
        * sample_idx
        * label
        * batch_size
    OPTIONAL INPUTS:
        * rows=8
        * cols=8
        * size=(20,15)
    GLOBAL INPUTS: None (matplotlib figure, printed to file)
    """

    fig = plt.figure(figsize=size)
    plt.title(CLASSES[int(label)])
    plt.axis('off')
    for n in range(0, batch_size):
        fig.add_subplot(rows, cols, n + 1)
        img = inp_batch[n]
        plt.imshow(img)
        plt.axis('off')

#-----------------------------------

def fit_knn_to_embeddings(model, X_train, ytrain, n_neighbors):
    """
    fit_knn_to_embeddings(model, X_train, ytrain, n_neighbors)
    INPUTS:
        * model [keras model]
        * X_train [list]
        * ytrain [list]
        * num_dim_use [int]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * knn [sklearn knn model]
    """
    embeddings = model.predict(X_train)
    embeddings = tf.nn.l2_normalize(embeddings, axis=-1)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(embeddings.numpy(), ytrain)
    return knn

#-----------------------------------
class EmbeddingModel(keras.Model):
    def train_step(self, data):
        # Note: Workaround for open issue, to be removed.
        if isinstance(data, tuple):
            data = data[0]
        anchors, positives = data[0], data[1]

        with tf.GradientTape() as tape:
            # Run both anchors and positives through model.
            anchor_embeddings = self(anchors, training=True)
            positive_embeddings = self(positives, training=True)

            # Calculate cosine similarity between anchors and positives. As they have
            # been normalised this is just the pair wise dot products.
            similarities = tf.einsum(
                "ae,pe->ap", anchor_embeddings, positive_embeddings
            )

            # Since we intend to use these as logits we scale them by a temperature.
            # This value would normally be chosen as a hyper parameter.
            temperature = 0.05 #0.1 ##0.2
            similarities /= temperature

            # We use these similarities as logits for a softmax. The labels for
            # this call are just the sequence [0, 1, 2, ..., num_classes] since we
            # want the main diagonal values, which correspond to the anchor/positive
            # pairs, to be high. This loss will move embeddings for the
            # anchor/positive pairs together and move all other pairs apart.
            sparse_labels = tf.range(num_classes)
            loss = self.compiled_loss(sparse_labels, similarities)

        # Calculate gradients and apply via optimizer.
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update and return metrics (specifically the one for the loss value).
        self.compiled_metrics.update_state(sparse_labels, similarities)
        return {m.name: m.result() for m in self.metrics}


##==================================================================

train_files = glob('All_Photos/TrainPhotosRecoded/*water*.jpg')[::2]
test_files = glob('All_Photos/TrainPhotosRecoded/*water*.jpg')[1::2]

CLASSES = ['no water', 'water'] #'not sure',
class_dict={'no_water':0,  'water':1} #'not_sure':1,
max_epochs = 100
num_batches = 500
lr = 5e-4
# n_neighbors = 3

# want a long feature vector for tsne mapping into just two dimensions for viz
num_embedding_dims = 512 #32 #100

size_vector = [600,400,200,100] #image sizes to use

num_classes = len(class_dict)
print(num_classes)


y_train = []
for f in train_files:
    y_train.append(class_dict[f.split(os.sep)[-1].split('_X_')[0]])

y_train = np.expand_dims(y_train,-1).astype('uint8')
y_train = np.squeeze(y_train)

class_idx_to_train_idxs = defaultdict(list)
for y_train_idx, y in enumerate(y_train):
    class_idx_to_train_idxs[y].append(y_train_idx)




####################################################
############ TRAINING
#################################################



# for height_width, num_embedding_dims in zip([800,600,400,200,150], [170,128,85,42,32]):
for height_width in size_vector: #800,600,

    print("image size: %i" % (height_width))
    print("embedding dims: %i" % (num_embedding_dims))

    x_train = np.zeros((len(train_files),height_width,height_width,3))
    for counter,f in enumerate(train_files):
        im = resize(imread(f), (height_width,height_width))
        x_train[counter]=standardize(im)
    x_train = x_train.astype("float32") #/ 255.0

    class AnchorPositivePairs(keras.utils.Sequence):
        def __init__(self, num_batchs):
            self.num_batchs = num_batchs

        def __len__(self):
            return self.num_batchs

        def __getitem__(self, _idx):
            x = np.empty((2, num_classes, height_width, height_width, 3), dtype=np.float32)
            for class_idx in range(num_classes):
                examples_for_class = class_idx_to_train_idxs[class_idx]
                anchor_idx = random.choice(examples_for_class)
                positive_idx = random.choice(examples_for_class)
                while positive_idx == anchor_idx:
                    positive_idx = random.choice(examples_for_class)
                x[0, class_idx] = x_train[anchor_idx]
                x[1, class_idx] = x_train[positive_idx]
            return x

    inputs = layers.Input(shape=(height_width, height_width, 3))
    x = layers.Conv2D(filters=16, kernel_size=3, strides=2, activation="relu")(inputs) #
    x = layers.Conv2D(filters=32, kernel_size=3, strides=2, activation="relu")(inputs) #
    x = layers.Conv2D(filters=64, kernel_size=3, strides=2, activation="relu")(inputs) #32
    x = layers.Conv2D(filters=128, kernel_size=3, strides=2, activation="relu")(x) #64
    x = layers.Conv2D(filters=256, kernel_size=3, strides=2, activation="relu")(x) #128
    x = layers.GlobalAveragePooling2D()(x)
    embeddings = layers.Dense(units=num_embedding_dims, activation=None)(x) #8
    embeddings = tf.nn.l2_normalize(embeddings, axis=-1)

    exec('model'+str(height_width)+' = EmbeddingModel(inputs, embeddings)')

    exec('model'+str(height_width)+'.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))')

    exec('history'+str(height_width)+'=model'+str(height_width)+'.fit(AnchorPositivePairs(num_batchs=num_batches), epochs=max_epochs)')

    del x_train, embeddings
    # just save the trained models


K.clear_session()


cols='rgbkmyc'
for counter,height_width in enumerate(size_vector):
    plt.plot(eval('history'+str(height_width)+'.history["loss"]'),cols[counter], label=str(height_width)+' px')
    plt.legend(fontsize=8)
plt.savefig('hist_'+'_'.join([str(s) for s in size_vector])+'.png', dpi=300, bbox_inches="tight")
plt.close()



####################################################
############ EVALUATION
#################################################

# pyramid version, stack outputs from multiple inputs each at different size
# embedding dimension same, so can stack depthwise and feed that as inputs to unsupervised model


E = []
for height_width in size_vector: #800,600,

    print("image size: %i" % (height_width))

    x_train = np.zeros((len(train_files),height_width,height_width,3))
    for counter,f in enumerate(train_files):
        im = resize(imread(f), (height_width,height_width))
        x_train[counter]=standardize(im)
    x_train = x_train.astype("float32") #/ 255.0

    exec('embeddings = model'+str(height_width)+'.predict(x_train)')
    del x_train
    E.append(tf.nn.l2_normalize(embeddings, axis=-1).numpy())
    del embeddings

K.clear_session()

# embeddings_train = np.zeros((E[0].shape[0],E[0].shape[1],len(E)))
# embeddings_train.shape
# for counter,e in enumerate(E):
#     embeddings_train[:,:,counter] = e


embeddings_train = np.hstack(E) #np.cast(E,'float32')


del E



# show examples per class
x_train = np.zeros((len(train_files),height_width,height_width,3))
for counter,f in enumerate(train_files):
    im = resize(imread(f), (height_width,height_width))
    x_train[counter]=standardize(im)
x_train = x_train.astype("float32") #/ 255.0

bs = 6
for class_idx in range(len(CLASSES)): # [0,1,2]:
  #show_one_class(class_idx=class_idx, bs=64)
  locs = np.where(y_train == class_idx)
  samples = locs[:][0]
  #random.shuffle(samples)
  samples = samples[:bs]
  print("Total number of {} (s) in the dataset: {}".format(CLASSES[class_idx], len(locs[:][0])))
  X_subset = x_train[samples]
  plot_one_class(X_subset, samples, class_idx, bs, CLASSES, rows=3, cols=2)
  # plt.show()
  plt.savefig( 'examples_class_samples_'+CLASSES[class_idx]+'.png', dpi=200, bbox_inches='tight')
  plt.close('all')



#prep test data

y_test = []
for f in test_files:
    y_test.append(class_dict[f.split(os.sep)[-1].split('_X_')[0]])
y_test = np.expand_dims(y_test,-1).astype('uint8')


## dim-red

tl=TSNE(n_components=2) #3)
embedding_tsne=tl.fit_transform(embeddings_train).astype('float32')
#
# colors = plt.cm.Blues(np.linspace(0, 1, num_classes))
#
# cmat = np.zeros((len(y_test),4))
# for k in range(len(y_test)):
#   cmat[k,:] = colors[y_test[k]]
#
# plt.figure(figsize=(10,10))
# im = plt.scatter(embedding_tsne[:,0], embedding_tsne[:,1], color=cmat, lw=.5, edgecolor='k')
# plt.show()

# n=32 emebeddings = does not separate the two classes
# n=512 emebeddings = does not separate the two classes


# kmeans = KMeans(init='k-means++', n_clusters=num_classes, n_init=10)
# kmeans.fit(embedding_tsne)
#
# cat = kmeans.predict(embedding_tsne)

ims = [resize(im, (64,64)) for im in x_train]

fig, ax = plt.subplots(figsize=(12,12))
artists = []
for xy, i in zip(embedding_tsne, ims):
    x0, y0 = xy
    img = OffsetImage(i, zoom=1.0)
    ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
    artists.append(ax.add_artist(ab))
ax.update_datalim(embedding_tsne[:,:2])
ax.autoscale()
ax.axis('tight')
ax.scatter(embedding_tsne[:,0], embedding_tsne[:,1], 20, y_train, zorder=10)

plt.savefig( 'tsne_vizimages.png', dpi=200, bbox_inches='tight')
plt.close('all')


x_test = np.zeros((len(test_files),height_width,height_width,3))
for counter,f in enumerate(test_files):
    im = resize(imread(f), (height_width,height_width))
    x_test[counter]=standardize(im)

x_test = x_test.astype("float32") #/ 255.0
y_test = np.squeeze(y_test)

class_idx_to_test_idxs = defaultdict(list)
for y_test_idx, y in enumerate(y_test):
    class_idx_to_test_idxs[y].append(y_test_idx)



E = []
for height_width in size_vector: #800,600,

    print("image size: %i" % (height_width))

    x_test = np.zeros((len(test_files),height_width,height_width,3))
    for counter,f in enumerate(test_files):
        im = resize(imread(f), (height_width,height_width))
        x_test[counter]=standardize(im)
    x_test = x_test.astype("float32") #/ 255.0

    exec('embeddings = model'+str(height_width)+'.predict(x_test)')
    del x_test
    E.append(tf.nn.l2_normalize(embeddings, axis=-1).numpy())
    del embeddings

K.clear_session()

#get embeddings for the test imagery

embeddings_test = np.hstack(E) #np.cast(E,'float32')

#make 3 knn models for k=7, k=9, and k=11
for n_neighbors in [7,9,11]:
    exec('knn'+str(n_neighbors)+'= KNeighborsClassifier(n_neighbors=n_neighbors)')
    exec('knn'+str(n_neighbors)+'.fit(embeddings_train, y_train)') #.numpy()

#     exec('y_pred'+str(n_neighbors)+' = knn'+str(n_neighbors)+'.predict_proba(embeddings_test)')
    K.clear_session()


# test kNN model
for n_neighbors in [7,9,11]:
    exec('score = knn'+str(n_neighbors)+'.score(embeddings_test, y_test)')
    print('KNN score: %f' % score) #knn3.score(embeddings_test, y_test)

##2-class, 400/200/100:
# image size: 400
# image size: 200
# image size: 100
#3KNN score: 0.942222
# 5KNN score: 0.942222
# 7KNN score: 0.951111

# KNN score: 0.920000
# KNN score: 0.924444
# KNN score: 0.924444


# print('KNN score: %f' % score) #knn3.score(embeddings_test, y_test)
# touse = len(x_test) #1000

# embeddings_test = model.predict(x_test[:touse])
# embeddings_test = tf.nn.l2_normalize(embeddings_test, axis=-1)
# # del X_test

# 800 px, 32 embed dim, KNN score: 0.797834
# 600 px, 32 embed dim, KNN score: 0.808664
# 400 px, 32 embed dim, KNN score: 0.815884
# 250 px, 32 embed dim,KNN score: 0.794224
# 150 px, 32 embed dim, : KNN score: 0.848375
# print('KNN score: %f' % knn3.score(embeddings_test[:,:num_dim_use], y_test[:touse]))
# del x_test, y_test


n_neighbors = 7
exec('y_pred = knn'+str(n_neighbors)+'.predict(embeddings_test)')


cm = confusion_matrix(y_test, y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# thres=0#.1
# cm[cm<thres] = 0

plt.figure(figsize=(8,8))
sns.heatmap(cm,
  annot=True,
  cmap = sns.cubehelix_palette(dark=0, light=1, as_cmap=True))

tick_marks = np.arange(len(CLASSES))+.5
plt.xticks(tick_marks, [c for c in CLASSES], rotation=90,fontsize=12)
plt.yticks(tick_marks, [c for c in CLASSES],rotation=0, fontsize=12)
plt.title('N = '+str(len(y_test)), fontsize=12)

plt.savefig('cm_nofiltbyprob.png', dpi=200, bbox_inches='tight')
plt.close()


## only 'certain' predictions
exec('y_pred = knn'+str(n_neighbors)+'.predict_proba(embeddings_test)')
K.clear_session()

y_prob = np.max(y_pred, axis=1)
y_pred = np.argmax(y_pred, axis=1)
ind = np.where(y_prob==1)[0]
print(len(ind))

cm = confusion_matrix(y_test[ind], y_pred[ind])
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# thres=0
# cm[cm<thres] = 0

plt.figure(figsize=(8,8))
sns.heatmap(cm,
  annot=True,
  cmap = sns.cubehelix_palette(dark=0, light=1, as_cmap=True))

tick_marks = np.arange(len(CLASSES))+.5
plt.xticks(tick_marks, [c for c in CLASSES], rotation=90,fontsize=12)
plt.yticks(tick_marks, [c for c in CLASSES],rotation=0, fontsize=12)
plt.title('N = '+str(len(ind)), fontsize=12)

plt.savefig('cm_filtbyprob.png', dpi=200, bbox_inches='tight')
plt.close()



####################################################
############ BASELINE COMPARISON
#################################################

num_components=100

# pca = PCA(n_components=num_components)
# reduced = pca.fit_transform(x_train.reshape(len(x_train),-1))
# print('Cumulative variance explained by {} principal components: {}'.format(num_components, np.sum(pca.explained_variance_ratio_)))
#
for n_neighbors in [7,9,11]:
    exec('pcaknn'+str(n_neighbors)+'= KNeighborsClassifier(n_neighbors=n_neighbors)')
    exec('pcaknn'+str(n_neighbors)+'.fit(reduced, y_train)') #.numpy()

#     exec('y_pred'+str(n_neighbors)+' = knn'+str(n_neighbors)+'.predict_proba(embeddings_test)')
    # K.clear_session()


for height_width in size_vector: #800,600,

    print("image size: %i" % (height_width))

    x_test = np.zeros((len(test_files),height_width,height_width,3))
    for counter,f in enumerate(test_files):
        im = resize(imread(f), (height_width,height_width))
        x_test[counter]=standardize(im)
    x_test = x_test.astype("float32") #/ 255.0

    pca = PCA(n_components=num_components)
    reduced_test = pca.fit_transform(x_test.reshape(len(x_test),-1))

    exec('pcaknn'+str(n_neighbors)+'= KNeighborsClassifier(n_neighbors=n_neighbors)')

    # test kNN model
    for n_neighbors in [7,9,11]:
        exec('score = pcaknn'+str(n_neighbors)+'.score(reduced_test, y_test)')
        print('pca-KNN score: %f' % score) #knn3.score(embeddings_test, y_test)

# image size: 400
# pca-KNN score: 0.715556
# pca-KNN score: 0.733333
# pca-KNN score: 0.742222
# image size: 200
# pca-KNN score: 0.742222
# pca-KNN score: 0.728889
# pca-KNN score: 0.751111
# image size: 100
# pca-KNN score: 0.800000
# pca-KNN score: 0.808889
# pca-KNN score: 0.800000


####################################################
############ APPLICATION ON UNKNOWN
#################################################

## read in 'not sure' images to classify

notsure_files = glob('All_Photos/TrainPhotosRecoded/*not*.jpg')

E = []
for height_width in size_vector: #800,600,

    print("image size: %i" % (height_width))

    x_test = np.zeros((len(notsure_files),height_width,height_width,3))
    for counter,f in enumerate(notsure_files):
        im = resize(imread(f), (height_width,height_width))
        x_test[counter]=standardize(im)
    x_test = x_test.astype("float32") #/ 255.0

    exec('embeddings = model'+str(height_width)+'.predict(x_test)')
    del x_test
    E.append(tf.nn.l2_normalize(embeddings, axis=-1).numpy())
    del embeddings

K.clear_session()

embeddings_notsure = np.hstack(E) #np.cast(E,'float32')


# better prediction probs with larger K
n_neighbors = 7
exec('y_pred = knn'+str(n_neighbors)+'.predict(embeddings_notsure)')
exec('y_prob = knn'+str(n_neighbors)+'.predict_proba(embeddings_notsure)')

for counter,f in enumerate(notsure_files):
    im = imread(f)
    plt.imshow(im); plt.axis('off')
    if y_pred[counter]==0:
        plt.title('Not Flooded (P='+str(y_prob[counter].max())[:5]+')')
    else:
        plt.title('Flooded (P='+str(y_prob[counter].max())[:5]+')')
    plt.savefig('not_sure_pred/'+f.split(os.sep)[-1].split('.jpg')[0]+'_pred.png')
    plt.close()









# fig = plt.figure(figsize=(12,12))
# ax = fig.add_subplot(projection='3d')
# ax.scatter(embedding_tsne[:,0], embedding_tsne[:,1], embedding_tsne[:,2], marker='o', color=cmat, alpha=1)

#
# In[67]:
#
#
# kmeans = KMeans(init='k-means++', n_clusters=num_classes, n_init=10)
# kmeans.fit(embedding_tsne)
# kmeans.fit(embeddings_train)


# In[68]:


# h = 1 #.02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# # Plot the decision boundary. For that, we will assign a color to each
# x_min, x_max = embedding_tsne[:, 0].min() - 1, embedding_tsne[:, 0].max() + 1
# y_min, y_max = embedding_tsne[:, 1].min() - 1, embedding_tsne[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# xx = xx.astype('float32')
# yy = yy.astype('float32')

# # Obtain labels for each point in mesh. Use last trained model.
# Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
# del yy

# Z = Z.reshape(xx.shape)
# del xx


# In[47]:


# # Put the result into a color plot
# plt.figure(figsize=(10,10))
# plt.clf()
# plt.imshow(Z, interpolation='nearest',
#            extent=(x_min, x_max, y_min, y_max), #extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#            cmap=plt.cm.Blues,
#            aspect='auto', origin='lower')

# plt.plot(embedding_tsne[:, 0], embedding_tsne[:, 1], 'k.', markersize=2)
# # Plot the centroids as a white X
# centroids = kmeans.cluster_centers_
# plt.scatter(centroids[:, 0], centroids[:, 1],
#             marker='x', s=169, linewidths=3,
#             color='m', zorder=10)
# plt.title('K-means clustering on the TSNE-reduced data\n'
#           'Centroids are marked with pink cross')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())

# for k in range(num_classes):
#    ind = np.where(kmeans.labels_ == k)[0]
#    plt.text(np.mean(embedding_tsne[ind, 0]), np.mean(embedding_tsne[ind, 1]), CLASSES[k] , color='k', fontsize=16)
