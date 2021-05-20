# FloodCam-WUSFeats
Flooded Road Detector using (weakly-/un-)supervised feature extraction, and uncoupled classification

### Supervised Deep Learning
In the companion repository, [NC-TrafficCams](https://github.com/FloodCamML/NCTrafficCameras), we have implemented a supervised deep neural network that couples feature extraction (using CNNs) explicitly with classes during the training process. In summsry:

* It uses mobilenet feature extraction with distillation head (max pool), and classifying head (dense layer with dropout and kernel regularization),
* It is retrained with data but feature extractor layers keep imagenet weights
* It has no supervised feature extraction when imagenet weights are used and frozen, but there is explicit supervised mapping of those features to classes by iteratively adjusting a model to do so
* It has enormous number of parameters, but only classification parameters tuned if used in transfer learning mode

### Weakly Supervised Deep Learning
Here I explore two alternatives to the above, that relax the level of supervision in the hope of creating a good alternative model; one that might even be more portable to sites and times and conditions outside those represented in training. It's also a good idea to use multiple indepedent models for ensemble or consensus approaches that might result in more robust predictions.

#### Model 1
supervised "uncoupled" classification from unsupervised feature extraction (baseline method)
* extract 100 principal components, then kNN classification
* lowest amount of supervision; no supervised feature extraction, no explicit coupling of features to classes (kNN functions like a lookup table)

#### Model 2
supervised "uncoupled" classification from features from weakly supervised feature extraction
* use convolution layers, global pooling, and dense layer with no activation to create embeddings from images
* weight the convolution layers using a loss function that just positions embeddings in embedding space such that embeddings are more similar to similar classes than different classes, so a very minimal bar
* (embeddings different from feature vectors becos dont change size with different sized images)
* medium supervision; weakly supervised feature extraction, no explicit coupling of features to classes (kNN functions like a lookup table)
* relative few parameters (~300,000), scales with convolution layer sizes
