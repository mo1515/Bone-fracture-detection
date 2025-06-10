from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import cv2
import numpy as np
from sklearn.cluster import KMeans
class SiftTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, sigma=0.8,nOctaveLayers=4,contrastThreshold=0.12):
        self.sigma = sigma          
        self.nOctaveLayers = nOctaveLayers
        self.contrastThreshold = contrastThreshold
        self.sift= None
    def fit(self,X,y=None):
        self.sift = cv2.SIFT_create(sigma=self.sigma, nOctaveLayers=self.nOctaveLayers, contrastThreshold=self.contrastThreshold)
        return self
    def transform(self, X):
        if self.sift is None:
            self.fit(X)
        descriptor = []
        for image in X:
            descriptor.append(self.sift.detectAndCompute(image,None)[-1])
        return descriptor
class BOVW(BaseEstimator,TransformerMixin):
    def __init__(self,k=100,random_state=42):
        self.k = k
        self.random_state = random_state
        self.kmeans = None
    def fit(self,X,y=None):
        filtered_descriptors = [np.array(desc) for desc in X if desc is not None]
        self.kmeans = KMeans(n_clusters=self.k, random_state=self.random_state)
        self.kmeans.fit(np.vstack(filtered_descriptors))
        return self
    def transform(self,X):
        if self.kmeans is None:
            self.fit(X)
        histograms = []
        for desc in X:
            hist =np.zeros(shape=[self.k],dtype=np.int16)
            if desc is not None:
                for descriptor in desc:
                    hist[self.kmeans.predict(descriptor.reshape(1,-1))]+=1
            histograms.append(hist)
        return np.array(histograms)

def preprocessingpipelinecreator(siftP : list,bovwP : list):
    if siftP is None:
        siftP={'sigma':0.8,'nOctaveLayers':4,'contrastThreshold':0.12}
    if bovwP is None:
        bovwP={'k':100,'random_state':42}
    
    return Pipeline([
        ('sift', SiftTransformer(**siftP)),
        ('bovw', BOVW(**bovwP)),
        ('scaler', StandardScaler())
    ])
        
    """pipeline=pl.preprocessingpipelinecreator(None,None)

train=pipeline.fit_transform(train_images)
test=pipeline.transform(test_images)
val=pipeline.transform(val_images)
    """