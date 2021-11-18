import torch
import torch.nn as nn
import torch.nn.functional as F



class RNN(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, out_size):
        super(RNN,self).__init__()
        self.rnn = nn.RNN(
                input_size = input_size,
                hidden_size= hidden_size,
                num_layers = num_layers,     
                batch_first= True 
        )
        self.fc = nn.Linear(hidden_size,out_size)
    def forward(self, x, h):
        #x   (time_step, batch_size, input_size)
        #h   (n_layer,  batch, hidden_size)
        #out (time_step, batch_size, hidden_size)
        out, _ = self.rnn(x, h)
        prediction = self.fc(out)
        return prediction, h

class one_lstm(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers,out_size):
        super(one_lstm, self).__init__()
        self.lstm = nn.LSTM(
                           input_size    = input_size,
                           hidden_size   = hidden_size,
                           num_layers    = num_layers  ,  
                           batch_first   = True,
)
        self.fc1 = nn.Linear(hidden_size, 1024)
        self.fc2 = nn.Linear(1024, out_size)
         
    def forward(self,x):
        out, _ = self.lstm(x)         
        out    = self.fc1(out)
        out    = self.fc2(out)
        return out


class bi_lstm(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers,out_size):
        super(bi_lstm, self).__init__()
        self.lstm = nn.LSTM(
                           input_size    = input_size,
                           hidden_size   = hidden_size,
                           num_layers    = num_layers  ,  
                           batch_first   = True,
                           bidirectional = True,
)
        self.fc1 = nn.Linear(hidden_size*2, 1024)
        self.fc2 = nn.Linear(1024, out_size)
         
    def forward(self,x):
        out,_   = self.lstm(x)         
        out    = self.fc1(out)
        out    = self.fc2(out)
        return out



#加入卷积的双向lstm
class bi_lstm_2(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers,out_size):
        super(bi_lstm, self).__init__()
        self.lstm = nn.LSTM(
                           input_size    = input_size,
                           hidden_size   = hidden_size,
                           num_layers    = num_layers  ,  
                           batch_first   = True,
                           bidirectional = True,
)       
        self.conv1 = nn.Conv2D(3,3)
        self.fc1   = nn.Linear(hidden_size*2, 1024)
        self.fc2   = nn.Linear(1024, out_size)
         
    def forward(self,x):
        out,_   = self.lstm(x)         
        out    = self.fc1(out)
        out    = self.fc2(out)
        return out




