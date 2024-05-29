import torch
import torch.nn as nn
import torch.nn.functional as F


###################  The different Layers used in the model  ###################  


#~##################  GRU Layer  ###################  

class GRULayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRULayer, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        
    def forward(self, x):
        full, last = self.gru(x)
        last = torch.transpose(last,0,1)
        return full, last
    

#~##################  Attention Layer  ###################    

class AttnLayer(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(AttnLayer, self).__init__()
        self.W1 = nn.Linear(input_shape, output_shape)
        self.W2 = nn.Linear(input_shape, output_shape)
        self.V = nn.Linear(input_shape, 1)
     
        
    def forward(self, full, last):
        scores = self.V(torch.tanh(self.W1(last) + self.W2(full)))
        attention_weights = F.softmax(scores, dim=1) 
        context_vector = attention_weights * full
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector


#~##################  Bilinear Layer  ################### 

class BilinearLayer(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(BilinearLayer, self).__init__()
        self.bilinear = nn.Bilinear(input_shape, input_shape, output_shape)
        
    def forward(self, x1, x2):
        return self.bilinear(x1, x2)


#~##################  LSTM Layer  ###################    

class LSTMLayer(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(LSTMLayer, self).__init__()
        self.lstm = nn.LSTM(input_shape, output_shape, batch_first=True)
        self.temporal_input = nn.Linear(output_shape + input_shape, output_shape + input_shape)
        
    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        temporal_input = torch.tanh(self.temporal_input(torch.cat((x, output), dim=2)))
        return temporal_input
    


#~##################  Temporal Attention Layer  ###################  

class TemporalAttn(nn.Module): 
    def __init__(self, input_shape):
        super(TemporalAttn, self).__init__()
        self.V1 = nn.Linear(input_shape , input_shape)
        self.V2 = nn.Linear(input_shape , 1)
        self.V3 = nn.Linear(input_shape , input_shape)
        self.V4 = nn.Linear(input_shape , 2)
        self.V5 = nn.Linear(2, 2)
        
     
     
    def forward(self, x):


        # Setting up the tensors of temporal information (G) and auxiliary predictions (Y)
        Y = torch.zeros(x.size(0), x.size(1) - 1, 2)
        G = torch.zeros(x.size(0), x.size(1) - 1, x.size(2))
        G_T = x[:, -1, :]
        for t in range(x.size(1) - 1):  
            G[:, t, :] = x[:, t, :]
        
        # Computing the information and dependency scores, and the attention score
        info_score = self.V2(torch.tanh(self.V1(G)))
        depend_score = torch.matmul(torch.tanh(self.V3(G)), torch.transpose(G_T.unsqueeze(1),1,2))
        attn_score = F.softmax(torch.mul(info_score , depend_score), dim=1)
        
        # Computing Auxiliary predictions (Y) for days t < T
        g_soft = F.softmax(self.V4(G), dim = 2)
        for t in range(x.size(1) - 1):
            Y[:, t, :] = g_soft[:,t,:]
        
        # Computing  prediction (Y) for day T
        Y_T = F.softmax(self.V5(torch.matmul(torch.transpose(attn_score, 1, 2), Y)), dim=2)
        
        return Y_T    