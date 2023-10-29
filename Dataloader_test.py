#!/usr/bin/env python
# coding: utf-8


import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
    
class time_series_decoder_paper(Dataset):
    """synthetic time series dataset from section 5.1"""
    
    def __init__(self,X,Y,transform=None):
        """
        Args:
            t0: previous t0 data points to predict from
            N: number of data points
            transform: any transformations to be applied to time series
        """
        N=np.size(X,0)
        
        # time points
        # self.x = torch.cat(N*[torch.arange(0,np.size(X,2)).type(torch.float).unsqueeze(0)])
        self.x = torch.cat(N*[torch.arange(0,np.size(X,1)).type(torch.float).unsqueeze(0)])
        self.transform = None
        X=torch.from_numpy(X).float()
        Y_tag=torch.from_numpy(Y).float()
        self.fx = X.type(torch.float)
        self.ytag = Y_tag.type(torch.float)
        # add noise
      
        
        self.masks = self._generate_square_subsequent_mask(np.size(X,1))
                
        
     
        
    def __len__(self):
        return len(self.fx)
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        
        sample = (self.x[idx,:],
                  self.fx[idx,:],
                  self.ytag[idx,:],
                  self.masks)
        
        if self.transform:
            sample=self.transform(sample)
            
        return sample
    
    def _generate_square_subsequent_mask(self,t0):
        mask = (np.full((t0, t0), -np.inf))
        for i in range(0,t0):
            mask[i,t0:] = 1 
        np.fill_diagonal(mask, 0)  # Set diagonal elements to 0
        mask[np.tril_indices(t0, k=-1)] = 0
        return mask