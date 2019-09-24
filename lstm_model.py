import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, args, embed_weights):
        super(LSTM, self).__init__()
        self.embed_size = args.embed_size
        self.max_length = args.max_length
        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_size
        self.n_layers = args.n_layers
        self.class_num = args.class_num

        self.embed = nn.Embedding.from_pretrained(Variable(torch.FloatTensor(embed_weights)), freeze=True)
        self.LSTM = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, num_layers=self.n_layers,batch_first=True)
        self.dropout = nn.Dropout(args.dropout)
        self.decoder = nn.Linear(self.hidden_size, self.class_num)

    def forward(self, x, x_length):
        x = self.embed(x)

        x = nn.utils.rnn.pack_padded_sequence(x, x_length, batch_first=True, enforce_sorted=False)
        h_0 = torch.zeros(self.n_layers, self.batch_size, self.hidden_size, requires_grad=True)
        c_0 = torch.zeros(self.n_layers, self.batch_size, self.hidden_size, requires_grad=True)
        if torch.cuda.is_available():
            h_0,c_0 = h_0.cuda(), c_0.cuda()
        x,(h_n,c_n) = self.LSTM(x, (h_0, c_0))

        prob = self.decoder(h_n[-1, :, :])
        return prob
