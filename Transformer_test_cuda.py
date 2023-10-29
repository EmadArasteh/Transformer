#!/usr/bin/env python
# coding: utf-8
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
import torch
import matplotlib.pyplot as plt
import causal_convolution_layer
import Dataloader_test
import h5py
import scipy.io

from sklearn.metrics import accuracy_score, cohen_kappa_score, balanced_accuracy_score, f1_score, confusion_matrix


# In[2]:
nbatch=8;
nchosen=40
nch=13
nepochs=1
Learning_rate=.0005
import math
from torch.utils.data import DataLoader
mat_file_path = '/hpc/umc_neonatology/earasteh/Python codes/Sleep Safe/Data/Features_Sleep.mat'
Itest = []
Itrain= []
# Open the .mat file using h5py
with h5py.File(mat_file_path, 'r') as file:
    # Access datasets within the file using dictionary-like syntax
    X = file['Feature_tot_agg'][:].T # Replace 'dataset_name_1' with the actual dataset name
    Y = file['Label_tot_agg'][:]-1  # Replace 'dataset_name_2' with the actual dataset name
    for column_test in file['index_test']:
        row_data_test = []
        for row_number in range(len(column_test)):            
            row_data_test .append(file[column_test[row_number]][:])   
        Itest.append(np.squeeze(np.array(row_data_test)))   
    for column_train in file['index_train']:
        row_data_train = []
        for row_number in range(len(column_train)):            
            row_data_train.append(file[column_train[row_number]][:])   
        Itrain.append(np.squeeze(np.array(row_data_train)))   
    

del file
X=X.astype(np.float64)
Y=Y.astype(np.float64)
# In[3]:


class TransformerTimeSeries(torch.nn.Module):
    """
    Time Series application of transformers based on paper
    
    causal_convolution_layer parameters:
        in_channels: the number of features per time point
        out_channels: the number of features outputted per time point
        kernel_size: k is the width of the 1-D sliding kernel
        
    nn.Transformer parameters:
        d_model: the size of the embedding vector (input)
    
    PositionalEncoding parameters:
        d_model: the size of the embedding vector (positional vector)
        dropout: the dropout to be used on the sum of positional+embedding vector
    
    """
    def __init__(self):
        super(TransformerTimeSeries,self).__init__()
        self.input_embedding = causal_convolution_layer.context_embedding(nch,256,9)
        self.positional_embedding = torch.nn.Embedding(1024,256)

        
        self.decode_layer = torch.nn.TransformerEncoderLayer(d_model=256,nhead=8)
        self.transformer_decoder = torch.nn.TransformerEncoder(self.decode_layer, num_layers=3)
        
        self.fc1 = torch.nn.Linear(256,1)
        
    def forward(self,x,y,attention_masks):
        z=y.permute(0,2,1)
        
        # ◘print(attention_masks)
        # input_embedding returns shape (Batch size,embedding size,sequence len) -> need (sequence len,Batch size,embedding_size)
        z_embedding = self.input_embedding(z).permute(2,0,1)
        

        # get my positional embeddings (Batch size, sequence_len, embedding_size) -> need (sequence len,Batch size,embedding_size)
        positional_embeddings = self.positional_embedding(x.type(torch.long)).permute(1,0,2)
       
    

        # print(z_embedding.shape)
        # print(positional_embeddings.shape)
        input_embedding = z_embedding+positional_embeddings
         

        transformer_embedding = self.transformer_decoder(input_embedding,attention_masks)

        output = self.fc1(transformer_embedding.permute(1, 0, 2).cuda())  # Move output to GPU
        # Apply softmax along the second dimension (axis=1)
        conv1d_layer = torch.nn.Conv1d(in_channels=600, out_channels=5, kernel_size=1).cuda()  # Move the convolution layer to GPU

        # print(output.shape)
        predicted_classes=conv1d_layer(output)



        return output,predicted_classes
        




# In[6]:


criterion = torch.nn.MSELoss()




# In[8]:


lr = Learning_rate # learning rate
epochs = nepochs


# In[10]:
def train_epoch(model,train_dl):
    model.train()
    train_loss = 0
    n = 0
    for step,(x,y,ytag,attention_masks) in enumerate(train_dl):
        # print(attention_masks.shape)
        optimizer.zero_grad()
        output_raw,predicted_classes_raw = model(x.cuda(),y.cuda(),attention_masks[0].cuda())
        criterion = torch.nn.CrossEntropyLoss()
        ytag=ytag.cuda()
        loss = criterion(predicted_classes_raw, ytag.clone().detach().long())
        # loss = criterion(output.squeeze()[:,(t0-1-10):(t0+24-1-10)],y[:,(t0-10):]) # missing data
        loss.backward()
        optimizer.step()
        train_loss += (loss.detach().cpu().item() * x.shape[0])
        n += x.shape[0]
    return train_loss/n




def test_epoch(model,test_dl):
    predictions = []
    observations = []
    with torch.no_grad():


        for step,(x,y,ytag,attention_masks) in enumerate(test_dl):
            output,predicted_classes_raw = model(x.cuda(),y.cuda(),attention_masks[0].cuda())
            predicted_classes = torch.argmax(predicted_classes_raw, dim=1)
            predictions.append(predicted_classes)
            observations.append(ytag)
            del predicted_classes_raw, ytag,output
            
    Concat_predict=torch.cat(predictions, dim=0)
    Total_predict = Concat_predict.detach().cpu().numpy()

    
    Concat_obs=torch.cat(observations, dim=0)
    Total_observation=Concat_obs.detach().cpu().numpy()   
    
    
    
    return Total_observation,Total_predict
    

# In[7]:
Conf_MATRIX_LOOP=[]
KAPPA_LOOP=[]
ACC_LOOP=[]
BACC_LOOP=[]
F1_LOOP=[]
OBSERVE_LOOP=[]
PREDICT_LOOP=[]
    
for i_train in range(len(Itrain)):
    
    model = TransformerTimeSeries().cuda()

    
    optimizer = torch.optim.Adam(model.parameters(nch), lr=lr)
    
    
    
    
    
    vec_train_raw=np.int64(Itrain[i_train])-1
    # vec_train=vec_train_raw[:nchosen]
    vec_train=vec_train_raw
    Y_train=Y[vec_train,:]
    X_train = np.transpose(np.squeeze(X[vec_train, :nch, :]), (0, 2, 1))
    train_dataset=Dataloader_test.time_series_decoder_paper(X_train,Y_train)
    train_dl = DataLoader(train_dataset,batch_size=nbatch,shuffle=True)
    
    
    
    vec_test_raw=np.int64(Itest[i_train])-1
    # vec_test=vec_test_raw[:nchosen]
    vec_test=vec_test_raw
    Y_test=Y[vec_test,:]
    X_test = np.transpose(np.squeeze(X[vec_test, :nch, :]), (0, 2, 1))
    test_dataset=Dataloader_test.time_series_decoder_paper(X_test,Y_test)
    test_dl = DataLoader(test_dataset,batch_size=nbatch,shuffle=True)
    
    train_epoch_loss = []
    eval_epoch_loss = []
    for e,epoch in enumerate(range(epochs)):
        train_loss = []
        eval_loss = []
        
        l_t = train_epoch(model,train_dl)
        train_loss.append(l_t)
        train_epoch_loss.append(np.mean(train_loss))
    
        if epoch%10==0:
            print("Train:{} Epoch {}: Train loss: {} ".format(i_train,e,np.mean(train_loss)))
        
    Total_observation,Total_predict = test_epoch(model,test_dl)    
    accuracy = accuracy_score(Total_observation, Total_predict)
    
    
    OBSERVE_LOOP.append(Total_observation)
    PREDICT_LOOP.append(Total_predict)
    # Calculate Kappa
    kappa = cohen_kappa_score(Total_observation, Total_predict)
    
    # Calculate Balanced Accuracy
    balanced_accuracy = balanced_accuracy_score(Total_observation, Total_predict)
    
    # Calculate F1 Score (micro average)
    f1 = f1_score(Total_observation, Total_predict, average='micro')
    
    # Calculate Confusion Matrix
    conf_matrix = confusion_matrix(Total_observation, Total_predict)
    
    print("Accuracy: {:.2f}".format(accuracy))
    print("Kappa: {:.2f}".format(kappa))
    print("Balanced Accuracy: {:.2f}".format(balanced_accuracy))
    print("F1 Score: {:.2f}".format(f1))
    print("Confusion Matrix:")
    print(conf_matrix)
    
    
    Conf_MATRIX_LOOP.append(conf_matrix)
    KAPPA_LOOP.append(kappa)
    ACC_LOOP.append(accuracy)
    BACC_LOOP.append(balanced_accuracy)
    F1_LOOP.append(f1)
    
    model_filename = f'model_iteration_{i_train}.pt'  # Specify the filename based on the iteration
    torch.save(model.state_dict(), model_filename)
    print(f'Model saved as {model_filename}')
    del model
    
    
# Find the maximum size of the confusion matrices
max_size = max(max(conf_matrix.shape) for conf_matrix in Conf_MATRIX_LOOP)

# Initialize an empty list to store padded confusion matrices
padded_conf_matrices = []

# Pad smaller matrices to make them 5x5
for conf_matrix in Conf_MATRIX_LOOP:
    padded_conf_matrix = np.zeros((max_size, max_size))
    padded_conf_matrix[:conf_matrix.shape[0], :conf_matrix.shape[1]] = conf_matrix
    padded_conf_matrices.append(padded_conf_matrix)

# Convert the list of padded matrices to a numpy array
padded_conf_matrices = np.array(padded_conf_matrices)


KAPPA_LOOP=np.array(KAPPA_LOOP)    
ACC_LOOP=np.array(ACC_LOOP)    
BACC_LOOP=np.array(BACC_LOOP)    
F1_LOOP=np.array(F1_LOOP)    
PREDICT_LOOP=np.array(PREDICT_LOOP)
OBSERVE_LOOP=np.array(OBSERVE_LOOP)

mat_data = {
    'Conf_MATRIX_LOOP': padded_conf_matrices,
    'KAPPA_LOOP': KAPPA_LOOP,
    'ACC_LOOP': ACC_LOOP,
    'BACC_LOOP': BACC_LOOP,
    'F1_LOOP': F1_LOOP,
    'OBSERVE_LOOP': OBSERVE_LOOP,
    'PREDICT_LOOP': PREDICT_LOOP
}

scipy.io.savemat('output_Transformer_Sleep.mat', mat_data)