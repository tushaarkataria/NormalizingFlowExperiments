from comet_ml import Experiment

from collections import OrderedDict
import os
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as torchd
import matplotlib.pyplot as plt
import argparse
from sklearn import datasets
from scipy import stats 
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

def sampleMask():
    u = np.random.uniform()
    if(u>0.5):
        mask = torch.from_numpy(np.array([0, 1]).astype(np.float32)).to(device=device, dtype=dtype)
    else: 
        mask = torch.from_numpy(np.array([1, 0]).astype(np.float32)).to(device=device, dtype=dtype)
    return mask


def getDistribution(args):
    if(args.dist=='twomoons'):
        x_a = datasets.make_moons(n_samples=args.inputSamples, noise=args.inputSampleNoise)[0].astype(np.float32)
    elif(args.dist=='circles'):
        x_a = datasets.make_circles(n_samples=args.inputSamples, noise=args.inputSampleNoise)[0].astype(np.float32)
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


class SNetwork(nn.Module):
     def __init__(self,inputDim,intermediateDimension,numberOfSLayers,activation):
        super(SNetwork,self).__init__()
        if activation == 'relu':
            act_layer = nn.ReLU()
        elif activation == 'tanh':
            act_layer = nn.ReLU()
        elif activation == 'lrelu':
            act_layer = nn.LeakyReLU()

        ## Declaring SNetwork
        moduleList =[]
        moduleList =[('FirstLayer',nn.Linear(inputDim,intermediateDimension)),('relu',act_layer)]
        for i in range(numberOfSLayers):
            moduleList.append(('intermediateLayers'+str(i),nn.Linear(intermediateDimension,intermediateDimension)))
            moduleList.append(('relu'+str(i),act_layer))
        moduleList.append(('finalLayer',nn.Linear(intermediateDimension,inputDim)))
        moduleList.append(('finalLayerTanh',nn.Tanh()))
        self.sNetwork = nn.Sequential(OrderedDict(moduleList))

     def forward(self,X):
         return self.sNetwork(X)





class TNetwork(nn.Module):
     def __init__(self,inputDim,intermediateDimension,numberOfTLayers,activation):
        super(TNetwork,self).__init__()
        if activation == 'relu':
            act_layer = nn.ReLU()
        elif activation == 'tanh':
            act_layer = nn.ReLU()
        elif activation == 'lrelu':
            act_layer = nn.LeakyReLU()

        ## Declaring SNetwork
        moduleList =[]
        moduleList =[('FirstLayer',nn.Linear(inputDim,intermediateDimension)),('relu',act_layer)]
        for i in range(numberOfTLayers):
            moduleList.append(('intermediateLayers'+str(i),nn.Linear(intermediateDimension,intermediateDimension)))
            moduleList.append(('relu'+str(i),act_layer))
        moduleList.append(('finalLayer',nn.Linear(intermediateDimension,inputDim)))

        self.tNetwork = nn.Sequential(OrderedDict(moduleList))
    
     def forward(self,X):
         return self.tNetwork(X)


class RealNVP2D(nn.Module):
    def __init__(self,inputDim,intermediateDimension,numberOfSLayers,numberOfTLayers,prior,activation,nof):
        super(RealNVP2D,self).__init__()
        self.nof = nof
        
        self.sarray = torch.nn.ModuleList([SNetwork(inputDim,intermediateDimension,numberOfTLayers,activation) for _ in range(nof)])
        self.tarray = torch.nn.ModuleList([TNetwork(inputDim,intermediateDimension,numberOfTLayers,activation) for _ in range(nof)])
        

        self.mask =[]
        for i in range(nof):
            self.mask.append(nn.Parameter(sampleMask(),requires_grad=False))
        self.prior = prior

    def forward(self,X):
        jacobianLoss =0
        priorLoss = 0
        for i in range(self.nof-1,-1,-1):
            z1 = self.mask[i]* X
            sOutput = self.sarray[i](z1)*(1-self.mask[i])
            tOutput = self.tarray[i](z1)*(1-self.mask[i])
            #zOuput = (1-self.mask[i])*(X* torch.exp(-sOutput)+tOutput) + z1
            zOutput = (1-self.mask[i])*(X-tOutput)* torch.exp(-sOutput)+ z1
            jacobianLoss += sOutput.sum(dim=1)
            X = zOutput
        priorloss     = self.prior.log_prob(X)
        return (jacobianLoss-priorloss).mean()

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        for i in range(self.nof):
            z1 = self.mask[i] * z
            sOutput = self.sarray[i](z1)*(1-self.mask[i])
            tOutput = self.tarray[i](z1)*(1-self.mask[i])
            zOutput = (1-self.mask[i])*(z* torch.exp(sOutput)+tOutput) + z1
            z = zOutput
        return zOutput







#class RealNVP2D(nn.Module):
#    def __init__(self,inputDim,intermediateDimension,numberOfSLayers,numberOfTLayers,prior,activation,nof):
#        super(RealNVP2D,self).__init__()
#        self.nof = nof
#        if activation == 'relu':
#            act_layer = nn.ReLU()
#        elif activation == 'tanh':
#            act_layer = nn.ReLU()
#        elif activation == 'lrelu':
#            act_layer = nn.LeakyReLU()
#
#        ## Declaring SNetwork
#        moduleList =[]
#        moduleList =[('FirstLayer',nn.Linear(inputDim,intermediateDimension)),('relu',act_layer)]
#        for i in range(numberOfSLayers):
#            moduleList.append(('intermediateLayers'+str(i),nn.Linear(intermediateDimension,intermediateDimension)))
#            moduleList.append(('relu'+str(i),act_layer))
#        moduleList.append(('finalLayer',nn.Linear(intermediateDimension,inputDim)))
#        moduleList.append(('finalLayerTanh',nn.Tanh()))
#        
#        sNetwork = nn.Sequential(OrderedDict(moduleList))
#         
#        ## Declaring T-Network
#        moduleList =[]
#        moduleList =[('FirstLayer',nn.Linear(inputDim,intermediateDimension)),('relu',act_layer)]
#        for i in range(numberOfSLayers):
#            moduleList.append(('intermediateLayers'+str(i),nn.Linear(intermediateDimension,intermediateDimension)))
#            moduleList.append(('relu'+str(i),act_layer))
#        moduleList.append(('finalLayer',nn.Linear(intermediateDimension,inputDim)))
#
#        tNetwork = nn.Sequential(OrderedDict(moduleList))
#
#        self.sarray = torch.nn.ModuleList([sNetwork for _ in range(nof)])
#        self.tarray = torch.nn.ModuleList([tNetwork for _ in range(nof)])
#        
#        self.mask =[]
#        for i in range(nof):
#            self.mask.append(nn.Parameter(sampleMask(),requires_grad=False))
#        self.mask = torch.from_numpy(np.array([[0, 1],[1,0]]*2).astype(np.float32))
#        print(self.mask)
#        self.prior = prior
#
#    def forward(self,X):
#        jacobianLoss =0
#        priorLoss = 0
#        for i in range(self.nof):
#            z1 = self.mask[i]* X
#            sOutput = self.sarray[i](z1)*(1-self.mask[i]) 
#            tOutput = self.tarray[i](z1)*(1-self.mask[i]) 
#            #zOuput = (1-self.mask[i])*(X* torch.exp(-sOutput)+tOutput) + z1
#            zOutput = (1-self.mask[i])*(X-tOutput)* torch.exp(-sOutput)+ z1
#            jacobianLoss += sOutput.sum(dim=1)
#            X = zOutput
#        priorloss     = self.prior.log_prob(X)
#        return (jacobianLoss-priorloss).mean()
#
#    def sample(self, batchSize):
#        z = self.prior.sample((batchSize, 1))
#        for i in range(self.nof):
#            z1 = self.mask[i] * z
#            sOutput = self.sarray[i](z1)*(1-self.mask[i])
#            tOutput = self.tarray[i](z1)*(1-self.mask[i])
#            zOutput = (1-self.mask[i])*(z* torch.exp(sOutput)+tOutput) + z1
#            z = zOutput
#        return zOutput

def main(args):
    prior = torch.distributions.MultivariateNormal(torch.zeros(2).to(device=device, dtype=dtype), torch.eye(2).to(device=device, dtype=dtype))
    moduleList = []
    flow =   RealNVP2D(2,args.intermediateDim,args.nS,args.nT,prior,args.act,args.nof)
    densityOrig = getDistribution(args).astype(np.float32)
    optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad==True], lr=args.learningRate)
    best_loss = 100000
    
    ## Prior
    z = np.random.multivariate_normal(np.zeros(2), np.eye(2), args.inputSamples*2)
    fig,ax =plt.subplots(4,4)
    fig.set_size_inches(20,20)
    ax[0][0].scatter(z[:, 0], z[:, 1],c='b')
    ax[0][0].title.set_text('Prior')
    
    x = densityOrig.astype(np.float32)
    ax[0][1].scatter(x[:, 0], x[:, 1], c='r')
    ax[0][1].title.set_text('Target Density')

    i=0
    j=2
    directoryName = os.path.join(str(args.dist),'act'+str(args.act),'nS'+str(args.nS),'nT'+str(args.nT),'interDim'+str(args.intermediateDim),'inputS'+str(args.inputSamples),'inputN'+str(args.inputSampleNoise),'nof'+str(args.nof)) 
    if not os.path.exists(directoryName):
           os.makedirs(directoryName) 
    for t in range(args.epoch):
        optimizer.zero_grad()
        density = torch.from_numpy(densityOrig).to(device=device, dtype=dtype)
        loss = 0
        flow.train()
        flow = flow.to(device=device, dtype=dtype)
        loss = flow(density)
        loss.backward(retain_graph=True)
        optimizer.step()
        if(loss.item() < best_loss):
                best_loss = loss.item()
                best_model = copy.deepcopy(flow.state_dict())
                torch.save(best_model,'./'+directoryName+'/bestmodel.pt') 
        if t % int(args.epoch/13) == 0:
            print('epoch %s:' % t, 'loss = %.6f' % loss)
            x = flow.sample(args.inputSamples*2).detach().cpu().numpy()
            ax[i][j].scatter(x[:, 0, 0], x[:, 0, 1], c='b')
            ax[i][j].title.set_text('flow Samples Epoch='+str(t) )
            j=j+1
            if(j>=4 and i<3):
                j=0
                i=i+1
            elif(i==3 and j==4):
                j=3
                i=3
    
    outputDirectory = './'
    saved_state_dict = torch.load(os.path.join(directoryName,'bestmodel.pt'))    
    flow.load_state_dict(saved_state_dict, strict=True)
    x = flow.sample(args.inputSamples*2).detach().cpu().numpy()
    ax[3][3].scatter(x[:, 0, 0], x[:, 0, 1], c='b')
    ax[3][3].title.set_text('Best Model Samples')

    fig.savefig(directoryName+'/image.png')
    ## Write a metric for matching distributions
    ## Currently Giving an error.
    #x1 = flow.sample(args.inputSamples).squeeze().detach().cpu().numpy()
    #print(x1.shape,densityOrig.shape)
    #temp = stats.ks_2samp(densityOrig, x1)
    #experiment.log_metric("ks2samp" ,temp)
    #plt.subplot(223)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('-lr', type=float,action="store", dest='learningRate', default=1e-4)
    parser.add_argument('-wd',type=float,action="store", dest='weightdecay', default=1e-4)
    parser.add_argument('-nof',type=int,action="store", dest='nof', default=4)
    parser.add_argument('-inputNoise',type=float,action="store", dest='inputSampleNoise', default=0.05)
    parser.add_argument('-inputSamples',type=int,action="store", dest='inputSamples', default=100)
    parser.add_argument('-nepoch',type=int, action="store", dest='epoch', default=1000)
    parser.add_argument('-interM',type=int, action="store", dest='intermediateDim', default=256)
    parser.add_argument('-nS',type=int, action="store", dest='nS', default=1)
    parser.add_argument('-nT',type=int, action="store", dest='nT', default=1)
    parser.add_argument('-act',type=str, action="store", dest='act', default='relu')
    parser.add_argument('-dist',type=str, action="store", dest='dist', default='twomoons')
    args = parser.parse_args()

    main(args)


