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
        # nn.init.xavier_uniform_(self.W1.weight)
        # nn.init.xavier_uniform_(self.W2.weight)
        # nn.init.xavier_uniform_(self.V.weight)
     
        
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
        temp = self.bilinear(x1, x2)
        out = F.relu(temp)
        return out       
    




#~##################  Graph Attention Layer  ################### 

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        
        self.W = nn.Linear(in_features, out_features, bias=False)
        nn.init.xavier_uniform_(self.W.weight.data, gain=1.414)
        
        self.a = nn.Linear(2*out_features, 1, bias=False)
        nn.init.xavier_uniform_(self.a.weight.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        device = input.device
        h =self.W(input)
        N = h.size()[0]

        a_input = torch.zeros(h.size(0), h.size(1), h.size(1), 2 * self.out_features, device=device)
        
        for i in range(15):
            for j in range(15):
                a_input[:,i,j,:] = torch.cat((h[:, i, :], h[:, j, :]), 1)
       
        e = self.leakyrelu(self.a(a_input)).squeeze(3)

        zero_vec = -9e15 * torch.ones_like(e, device=device)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = F.elu(torch.matmul(attention, h))

        return h_prime
        
        
        

#~################## Multihead Attention  ###################   

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, in_features, out_features, dropout, alpha):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.graph_attentions = nn.ModuleList([GraphAttentionLayer(in_features, out_features, dropout, alpha) 
                                 for _ in range(self.n_heads)])
        
        
    def forward(self, input, adj):
        device = input.device
        h_prime = torch.zeros(self.n_heads, input.size(0), input.size(1), input.size(2), device=device)

        for i, attn  in enumerate(self.graph_attentions):
            h_prime[i, :, :, :] = attn(input, adj)

        return torch.cat([h_prime[i,:,:,:] for i in range(self.n_heads)], dim=2)    
        
        
        
