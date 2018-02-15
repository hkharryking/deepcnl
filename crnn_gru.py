from crnn_stock.crnn import CRNN
from torch import nn
import torch
from torch.autograd import Variable

class CRNN_GRU(CRNN):
    """
    A mixed deep learning framework with Convolution and GRU
    """

    def get_code(self):
        return 'CRNN_GRU'

    def __init__(self,feature_num, filters_num, window, ticker_num, hidden_unit_num,hidden_layer_num,dropout_ratio):
        super(CRNN, self).__init__()
        self.hidden_layer_num = hidden_layer_num
        self.filters_num = filters_num
        self.hidden_unit_num = hidden_unit_num
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=feature_num * 2,
                      out_channels=filters_num,
                      kernel_size=window,
                      stride=1,
                      padding=0),
            nn.ReLU(True),
            #            nn.MaxPool1d(WINDOW),
            #            nn.Conv1d(in_channels=6,
            #                       out_channels=1,
            #                       kernel_size=WINDOW,
            #                       stride=1,
            #                       padding = 0),
            #            #nn.ReLU(True),
            # nn.MaxPool1d(WINDOW)4
        )
        self.bn = nn.BatchNorm1d(filters_num)
        self.pool = nn.MaxPool1d(int(ticker_num*(ticker_num-1)/2))
        self.rnn = nn.GRU(
                                input_size = int(ticker_num*(ticker_num-1)/2),
                                hidden_size = hidden_unit_num,     #  hidden unit
                                num_layers = hidden_layer_num,
                                dropout=dropout_ratio,
                                batch_first=True
                )
        #self.batchnormal=nn.BatchNorm1d(filters_num)

        self.line = nn.Linear(filters_num, 1)
        torch.nn.init.xavier_uniform(self.line.weight)
        self.hidden = self.init_hidden()


    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        h0 = Variable(torch.zeros(self.hidden_layer_num, self.filters_num, self.hidden_unit_num)).cuda().float()
        h0 = torch.nn.init.xavier_uniform(h0)
        return h0 #rnn GRU