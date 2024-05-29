import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers import AttnLayer, GRULayer, BilinearLayer, LSTMLayer, TemporalAttn


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
        ct = torch.cat(cvs, 1)

        return ct
    

#~################## Price Data Encoder  ###################      

class PriceEncoder(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(PriceEncoder, self).__init__()
        self.gru_layer = GRULayer(input_shape, output_shape)
        

    def forward(self, x):
        qt, last = self.gru_layer(x)
        return qt
    


#~##################  Concatenation Processor  ###################    

class ConcatProcessor(nn.Module):
    def __init__(self, text_input_shape,  price_input_shape, hidden_size):
        super(ConcatProcessor, self).__init__()
        self.SMEncoder = SMEncoder(text_input_shape, hidden_size)  #~##
        self.PriceEncoder = PriceEncoder(price_input_shape, hidden_size) #~##
        self.bilinear = BilinearLayer(hidden_size, hidden_size) #~##
        self.lstm_layer = LSTMLayer(hidden_size, hidden_size) #~##
        self.temporal_attn = TemporalAttn(int(hidden_size * 2)) #~##
        
    def forward(self, s, p):

        # Social Media Information Encoder
        ct = self.SMEncoder(s)
        # Price Data Encoder
        qt = self.PriceEncoder(p)
        # Bilinear Layer
        bilinear = self.bilinear(ct, qt)
        # LSTM Layer
        lstm = self.lstm_layer(bilinear)
        # Temporal Attention Layer
        res = self.temporal_attn(lstm)
        res = res.squeeze()


        return res