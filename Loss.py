import sys
sys.path.append('/home/john/caffe-master/python/')

import caffe
import numpy as np
from skimage import filter
from skimage.morphology import dilation, disk
from numpy import linalg as LA

class WeightedSoftmaxLossLayer(caffe.Layer):
    """
    Compute the Softmax Loss in the same manner but use the skeletal loss as weights
    """
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need 2 inputs to compute distance.")
 
    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].num != bottom[1].num:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # weights matrix
        self.mask = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)
	# ratio for imbalanced data
	self.ratio = 0.5
 
    def forward(self, bottom, top):        
        
        # run softmax() layer 
        score = np.copy(bottom[0].data)
        
        temp = np.maximum(score[0,0,:,:], score[0,1,:,:])
        score[0,0,:,:] -= temp
        score[0,1,:,:] -= temp
        
        prob = np.exp(score)
        
        temp = prob[0,0,:,:] + prob[0,1,:,:]
        prob[0,0,:,:] /= temp
        prob[0,1,:,:] /= temp
        
        # generate weights matrix
        label = np.copy(bottom[1].data)

	# calcualte self.ratio
	self.ratio = np.count_nonzero(label==1) * 1.0 / np.count_nonzero(label!=255)

        temp = np.copy(label)
        temp[...] = 0
        temp[np.where(label==0)] = self.ratio
        temp[np.where(label==255)] = 0
        self.mask[0,0,:,:] = np.copy(temp)
        temp[...] = 0
        temp[np.where(label>0)] = 1.0 - self.ratio
        temp[np.where(label==255)] = 0
        self.mask[0,1,:,:] = np.copy(temp)
        count = np.count_nonzero(self.mask)

        weights = np.copy(self.mask)
        
        # calculate loss
        probs = np.copy(prob)
        probs[np.where(probs<1.175494e-38)] = 1.175494e-38
        logprob = -np.log(probs)
        
        data_loss = np.sum(weights*logprob) *1.0 / count
        
        self.diff[...] = np.copy(prob)
        
        top[0].data[...] = np.copy(data_loss)
 
    def backward(self, top, propagate_down, bottom):
        
        delta = np.copy(self.diff[...])
        
        count = np.count_nonzero(self.mask)
        
        delta[np.where(self.mask>0)] -= 1
        
        # generate pixel-wise matrix
        label = np.copy(bottom[1].data)
        weights = np.copy(bottom[0].data)
        temp = np.copy(label)
        temp[...] = 0
        temp[np.where(label==0)] = self.ratio
        temp[np.where(label>0)] = 1.0 - self.ratio
        temp[np.where(label==255)] = 0
        weights[0,0,:,:] = np.copy(temp)
        weights[0,1,:,:] = np.copy(temp)
        
        delta *= weights
        bottom[0].diff[...] = delta * 1.0 / count

class ContourLossLayer(caffe.Layer):
    """
    Compute the Softmax Loss in the same manner but use the skeletal loss as weights
    """
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 4:
            raise Exception("Need 4 inputs to compute distance.")
 
    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].num != bottom[1].num:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # weights matrix
        self.mask = np.zeros_like(bottom[0].data, dtype=np.float32)
        # similarity matrix
        self.contourmask = np.ones_like(bottom[1].data, dtype=np.float32)
	# weights matrix
        self.Weights = np.ones_like(bottom[1].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)
	# ratio for imbalanced data
	self.ratio = 0.5
 
    def forward(self, bottom, top):        
        
        # run softmax() layer 
        score = np.copy(bottom[0].data)
        
        temp = np.maximum(score[0,0,:,:], score[0,1,:,:])
        score[0,0,:,:] -= temp
        score[0,1,:,:] -= temp
        
        prob = np.exp(score)
        
        temp = prob[0,0,:,:] + prob[0,1,:,:]
        prob[0,0,:,:] /= temp
        prob[0,1,:,:] /= temp
        
        # calcl skeletal loss
        img = np.copy(prob[0,1,:,:])
        img[np.where(img<0.5)] = 0
        img[np.where(img>0)] = 1
        
        label = np.copy(bottom[1].data[0,0,:,:])
        IDMask = np.copy(bottom[2].data[0,0,:,:])

	w = np.copy(bottom[3].data[0,0,:,:])
	w[np.where(w>30)] = 0
	self.Weights = 1.0 / (1.0 + np.exp(w - 15.0))
	self.Weights[np.where(w==0)] = 0
	
        self.contourmask[0,0,:,:] = 1.0 + 4.0 * (1.0 - self.ContourLoss(img, label, IDMask))

        # generate pixel-wise matrix
        label = np.copy(bottom[1].data)
	Range = np.copy(bottom[2].data)
	#label[np.where(Range==0)] = 0
	
	# calcualte self.ratio
	self.ratio = np.count_nonzero(label==1) * 1.0 / np.count_nonzero(label!=255)

        temp = np.copy(label)
        temp[...] = 0
        temp[np.where(label==0)] = self.ratio
        self.mask[0,0,:,:] = np.copy(temp)
        temp[...] = 0
        temp[np.where(label!=0)] = 1 - self.ratio
        self.mask[0,1,:,:] = np.copy(temp)
        count = np.count_nonzero(self.mask)
        
        #weights: combination of self.mask and self.skelmask
        weights = self.mask * self.contourmask * (1.0 + 6.0 * self.Weights) 
        
        # calculate loss
        probs = np.copy(prob)
        probs[np.where(probs<1.175494e-38)] = 1.175494e-38
        logprob = -np.log(probs)
        
        data_loss = np.sum(weights*logprob) * 1.0 / count
        
        self.diff[...] = np.copy(prob)
        
        top[0].data[...] = np.copy(data_loss)
 
    def backward(self, top, propagate_down, bottom):
        
        delta = np.copy(self.diff[...])
        
        count = np.count_nonzero(self.mask)
        
        delta[np.where(self.mask>0)] -= 1
        
        # generate pixel-wise matrix
	label = np.copy(bottom[1].data)
	Range = np.copy(bottom[2].data)
	#label[np.where(Range==0)] = 0
	Weights = np.copy(bottom[3].data[0,0,:,:])

        mask = np.copy(bottom[0].data)
        temp = np.copy(label)
        temp[...] = 0
        temp[np.where(label==0)] = self.ratio
	temp[np.where(label!=0)] = 1 - self.ratio
        mask[0,0,:,:] = np.copy(temp)
        mask[0,1,:,:] = np.copy(temp)
        
        #weights: combination of self.mask and self.skelmask
        weights = mask * self.contourmask * (1.0 + 6.0 * self.Weights) 
        
        delta *= weights
        bottom[0].diff[...] = delta * 1.0 / count
        
    def ContourLoss(self, img, label, IDMask):

        contour = filter.canny(img)

	selem = disk(5)
	outlier = dilation(contour, selem)
	outlier[np.where(np.absolute(IDMask)>0)] = 0	

        contour = contour.astype(float)
        contour *= IDMask
        
        Similarity = np.ones_like(IDMask, dtype=np.float)
	Similarity[np.where(outlier>0)] = -0.5
       
        if np.amax(IDMask) > 100000:
            raise Exception("Wrong in IDMask!")

        LocationsSrc = {index: np.where(np.absolute(contour)==index+1) for index in range(np.amax(IDMask))}
        LocationsRef = {index: np.where(IDMask==-(index+1)) for index in range(np.amax(IDMask))}

        for index in range(np.amax(IDMask)):
            SrcX = LocationsSrc[index][0]
            SrcY = LocationsSrc[index][1]
            RefX = LocationsRef[index][0]
            RefY = LocationsRef[index][1]

	    ContourSimilarity = 0
            if np.size(SrcX) > 0.9 * np.size(RefX) and np.size(RefX) > 10 and np.size(SrcX) < 1.2 * np.size(RefX):
                if np.size(np.unique(SrcX)) > np.size(np.unique(SrcY)):
                    VecSrc = np.polyfit(SrcX, SrcY, 3)[:3]
                    VecRef = np.polyfit(RefX, RefY, 3)[:3]
                else:
                    VecSrc = np.polyfit(SrcY, SrcX, 3)[:3]
                    VecRef = np.polyfit(RefY, RefX, 3)[:3]

                ContourSimilarity = np.absolute(np.inner(VecSrc, VecRef)) / (LA.norm(VecSrc) + 1e-10) / (LA.norm(VecRef) + 1e-10)
		
            Similarity[np.where(np.absolute(IDMask)==index+1)] = ContourSimilarity

        return Similarity
