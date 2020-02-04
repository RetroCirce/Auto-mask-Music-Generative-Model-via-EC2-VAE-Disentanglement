import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


class MaskNN(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, time_steps):
        super(MaskNN, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.time_steps = time_steps
        self.LTSM_e1 = nn.LSTM(input_size = input_dims, hidden_size = hidden_dims,
                        batch_first = True,bidirectional = True) 
        self.dropout_1 = nn.Dropout(p = 0.5)
        self.LTSM_e2 = nn.LSTM(input_size = hidden_dims * 2, hidden_size = hidden_dims,
                        batch_first = True,bidirectional = True)
        self.dropout_2 = nn.Dropout(p = 0.5)
        self.LTSM_e3 = nn.LSTM(input_size = hidden_dims * 2, hidden_size = hidden_dims,
                        batch_first = True,bidirectional = True)
        self.dropout_3 = nn.Dropout(p = 0.5)
        self.LTSM_e4 = nn.LSTM(input_size = hidden_dims * 2, hidden_size = hidden_dims,
                        batch_first = True,bidirectional = True)
        self.LTSM_d1 = nn.LSTM(input_size = hidden_dims * 2, hidden_size = hidden_dims,
                        batch_first = True,bidirectional = True)
        self.dropout_4 = nn.Dropout(p = 0.5)
        self.LTSM_d2 = nn.LSTM(input_size = hidden_dims * 2, hidden_size = hidden_dims,
                        batch_first = True,bidirectional = True)
        self.dropout_5 = nn.Dropout(p = 0.5)
        self.LTSM_d3 = nn.LSTM(input_size = hidden_dims * 2, hidden_size = hidden_dims,
                        batch_first = True,bidirectional = True)
        self.dropout_6 = nn.Dropout(p = 0.5)
        self.LTSM_d4 = nn.LSTM(input_size = hidden_dims * 2, hidden_size = hidden_dims,
                        batch_first = True,bidirectional = True)    
        self.fc = nn.Linear(hidden_dims * 2, output_dims)
        
    def encoder(self, x):
        model_x, _ = self.LTSM_e1(x)
        model_x = self.dropout_1(model_x)
        pre_x = model_x
        model_x, _ = self.LTSM_e2(model_x)
        model_x = self.dropout_2(model_x)
        model_x += pre_x # 1-d shortcut
        pre_x = model_x
        model_x, _ = self.LTSM_e3(model_x)
        model_x = self.dropout_3(model_x)
        model_x += pre_x
        pre_x = model_x
        model_x, _ = self.LTSM_e4(model_x)
        model_x = model_x + pre_x
        return model_x
    def attentioner(self, x):
        # to be continued
        return True
    def decoder(self,x):
        model_x, _ = self.LTSM_d1(x)
        model_x = self.dropout_4(model_x)
        pre_x = model_x
        model_x, _ = self.LTSM_d2(model_x)
        model_x = self.dropout_5(model_x)
        model_x += pre_x # 1-d shortcut
        pre_x = model_x
        model_x, _ = self.LTSM_d3(model_x)
        model_x = self.dropout_6(model_x)
        model_x += pre_x
        pre_x = model_x
        model_x, _ = self.LTSM_d4(model_x)
        model_x = model_x + pre_x
        return model_x
    def forward(self, x):
        self.e_out = self.encoder(x)
        self.d_out = self.decoder(self.e_out)
        model_out = self.fc(self.d_out)
        self.model_out = model_out
        return self.model_out



