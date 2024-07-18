import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from layers import AttnLayer, GRULayer, BilinearLayer, GraphAttentionLayer, MultiHeadAttention


#~################## Social Media Information Encoder  ###################  

class SMEncoder(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(SMEncoder, self).__init__()
        self.gru_intra = GRULayer(input_shape, output_shape)
        self.attn_intra = AttnLayer(output_shape, output_shape)
        self.gru_inter = GRULayer(output_shape, output_shape)
        self.attn_inter = AttnLayer(output_shape, output_shape)

    def forward(self, x):
        
        # Rearrange the input so we can iterate over days
        text_t = []
        for i in range(x.size(1)):
            text_t.append(x[:, i, :])
        
        # Intra Day Representation
        cvs = []
        for i in range(x.size(1)):
            full1, last1 = self.gru_intra(text_t[i])
            context_vector = self.attn_intra(full1, last1)
            cv1 = torch.unsqueeze(context_vector, 1)
            cvs.append(cv1)
        c1 = torch.cat(cvs, 1)

        # Inter Day Representation
        full2, last2 = self.gru_inter(c1)
        ct = self.attn_inter(full2, last2)

        return ct
    

#~################## Price Data Encoder  ###################      

class PriceEncoder(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(PriceEncoder, self).__init__()
        self.gru_layer = GRULayer(input_shape, output_shape)
        self.attn_layer = AttnLayer(output_shape, output_shape)
        

    def forward(self, x):
        full, last = self.gru_layer(x)
        qt = self.attn_layer(full, last)
        return qt
    

#~##################  Graph Attention Network  ###################      

class GAT(nn.Module):
    def __init__(self, text_input_shape,  price_input_shape, num_stocks, hidden_size, 
                 num_heads, num_classes, dropout, alpha,training=True):
        super(GAT, self).__init__()
        self.num_stocks = num_stocks
        self.dropout = dropout
        self.training = training
        self.num_classes = num_classes
        self.SMEncoder = SMEncoder(text_input_shape, hidden_size)  #~##
        self.PriceEncoder = PriceEncoder(price_input_shape, hidden_size) #~##
        self.bilinear = BilinearLayer(hidden_size, hidden_size) #~##
        self.hidden_size = hidden_size
        self.graph_attention1 = MultiHeadAttention(num_heads, hidden_size, hidden_size, dropout, alpha)
        self.graph_attention2 = GraphAttentionLayer(num_heads * hidden_size, num_classes, dropout, alpha)
        self.linearx = nn.ModuleList([nn.Linear(hidden_size, num_classes) for _ in range(num_stocks)])
        


    def forward(self, s, p, adj):
        device = s.device
        # tensor to hold bimodal representation of all stocks
        X = torch.zeros(s.size(0), self.num_stocks, self.hidden_size, device=device)
        out_1 = torch.zeros(s.size(0), self.num_stocks, self.num_classes, device=device)


        for i in range(self.num_stocks):
            # Social Media Information Encoder
            ct = self.SMEncoder(s[:, :, :, :, i]).to(device)
            # Price Data Encoder
            qt = self.PriceEncoder(p[:, :, :, i]).to(device)
            # Bilinear Layer
            bilinear = self.bilinear(ct, qt)
            X[:, i, :] = bilinear
            out_1[:,i,:] = F.tanh(self.linearx[i](bilinear))
        
        
        X = F.dropout(X, self.dropout, training=self.training)
        # Graph Attention Layer 1
        X = self.graph_attention1(X, adj)
        X = F.dropout(X, self.dropout, training=self.training)
        # Graph Attention Layer 2
        X = self.graph_attention2(X, adj)
        X = X + out_1    
        X = torch.transpose(X,2,1)
        
        
        return X   
    



























