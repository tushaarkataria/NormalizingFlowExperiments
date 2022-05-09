from comet_ml import Experiment

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as torchd
import matplotlib.pyplot as plt
import argparse
from sklearn import datasets
 
#import albumentations as A
#import albumentations.augmentations.functional as F
#from albumentations.pytorch import ToTensorV2
from numpy import pi

import copy
import sys

device = torch.device('cuda')
dtype = torch.float32

experiment = Experiment(
    api_key="lv2ScJmA1lpzTzBFWzbXh6fQw",
    project_name="real-nvp-project",
    workspace="machinelearningproject",
)

def getDistribution(args):
    if(args.dist=='twomoons'):
        x_a = datasets.make_moons(n_samples=args.inputSamples, noise=args.inputSampleNoise)[0].astype(np.float32)
    elif(args.dist=='circles'):
        x_a = datasets.make_circles(n_samples=args.inputSamples, noise=args.inputSampledNoise)[0].astype(np.float32)
    elif(args.dist=='spiral2'):
        N = args.inputSamples
        theta = np.sqrt(np.random.rand(N))*2*pi # np.linspace(0,2*pi,100)
        r_a = 2*theta + pi
        data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
        if(args.inputSampleNoise>0.0):
            x_a = data_a + np.random.randn(N,2)
        else:
            x_a = data_a
    elif(args.dist=='spiral4'):
        N = args.inputSamples
        theta = np.sqrt(np.random.rand(N))*4*pi # np.linspace(0,2*pi,100)
        r_a = 2*theta + pi
        data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
        if(args.inputSampleNoise>0.0):
            x_a = data_a + np.random.randn(N,2)
        else:
            x_a = data_a
    return x_a 

class RealNVP2D(nn.Module):
    def __init__(self,inputDim,intermediateDimension,numberOfSLayers,numberOfTLayers,prior,activation,mask):
        super(RealNVP2D,self).__init__()
        if activation == 'relu':
            act_layer = nn.ReLU()
        elif activation == 'tanh':
            act_layer = nn.ReLU()
        elif activation == 'lrelu':
            act_layer = nn.LeakyReLU()

        ## Declaring SNetwork
        moduleList =[]
        print(moduleList)
        moduleList =[('FirstLayer',nn.Linear(inputDim,intermediateDimension)),('relu',act_layer)]
        for i in range(numberOfSLayers):
            moduleList.append(('intermediateLayers'+str(i),nn.Linear(intermediateDimension,intermediateDimension)))
            moduleList.append(('relu'+str(i),act_layer))
        print(moduleList)
        moduleList.append(('finalLayer',nn.Linear(intermediateDimension,inputDim)))
        moduleList.append(('finalLayerTanh',nn.Tanh()))
        print(moduleList)
        self.sNetwork = nn.Sequential(OrderedDict(moduleList))
        
        ## Declaring T-Network
        moduleList =[]
        moduleList =[('FirstLayer',nn.Linear(inputDim,intermediateDimension)),('relu',act_layer)]
        for i in range(numberOfSLayers):
            moduleList.append(('intermediateLayers'+str(i),nn.Linear(intermediateDimension,intermediateDimension)))
            moduleList.append(('relu'+str(i),act_layer))
        moduleList.append(('finalLayer',nn.Linear(intermediateDimension,inputDim)))

        self.tNetwork = nn.Sequential(OrderedDict(moduleList))


        self.prior = prior
        self.mask = mask

    def forward(self,X):
        log_det_J = X.new_zeros(X.shape[0])
        z1 = self.mask * X
        sOutput = self.sNetwork(z1) *(1-self.mask)
        tOutput = self.tNetwork(z1) *(1-self.mask)
        #zOuput = (1-self.mask)*(X* torch.exp(-sOutput)+tOutput) + z1
        zOuput = (1-self.mask)*(X-tOutput)*torch.exp(-sOutput) + z1
        log_det_J -= sOutput.sum(dim=1)
        loss = self.prior.log_prob(zOuput) + log_det_J
        return loss


    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        z1 = self.mask * z
        sOutput = self.sNetwork(z1) 
        tOutput = self.tNetwork(z1) 
        zOuput = (1-self.mask)*(z* torch.exp(sOutput)+tOutput) + z1
        return zOuput

def main(args):
    mask = torch.from_numpy(np.array([0, 1]).astype(np.float32))
    prior = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
    flow = RealNVP2D(2,args.intermediateDim,args.nS,args.nT,prior,args.act,mask)
    
    density = getDistribution(args) 
    optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad==True], lr=args.learningRate)
    for t in range(args.epoch):
        density = density.astype(np.float32)
        loss = -flow(torch.from_numpy(density)).mean()
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    
        if t % 500 == 0:
            print('iter %s:' % t, 'loss = %.3f' % loss)
    
    z = np.random.multivariate_normal(np.zeros(2), np.eye(2), 1000)
    plt.subplot(221)
    plt.scatter(z[:, 0], z[:, 1])
    plt.title(r'$z \sim p(z)$ (Base Density)')
    
    plt.subplot(222)
    x = density.astype(np.float32)
    plt.scatter(x[:, 0], x[:, 1], c='r')
    plt.title(r'$X \sim p(X)$ (Target Density) ')
    
    plt.subplot(223)
    x = flow.sample(1000).detach().numpy()
    plt.scatter(x[:, 0, 0], x[:, 0, 1], c='r')
    plt.title(r'$X = g(z)$ (Predicted Density after Flow Tranformation)')
    plt.savefig('image.png')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('-lr', type=float,action="store", dest='learningRate', default=1e-4)
    parser.add_argument('-wd',type=float,action="store", dest='weightdecay', default=1e-4)
    parser.add_argument('-inputNoise',type=float,action="store", dest='inputSampleNoise', default=0.05)
    parser.add_argument('-inputSamples',type=int,action="store", dest='inputSamples', default=200)
    parser.add_argument('-nepoch',type=int, action="store", dest='epoch', default=10000)
    parser.add_argument('-interM',type=int, action="store", dest='intermediateDim', default=512)
    parser.add_argument('-nS',type=int, action="store", dest='nS', default=5)
    parser.add_argument('-nT',type=int, action="store", dest='nT', default=5)
    parser.add_argument('-act',type=str, action="store", dest='act', default='relu')
    parser.add_argument('-dist',type=str, action="store", dest='dist', default='twomoons')
    args = parser.parse_args()

    main(args)


