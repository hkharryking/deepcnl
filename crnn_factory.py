from crnn_stock.crnn_rnn import CRNN_RNN
from crnn_stock.crnn_lstm import CRNN_LSTM
from crnn_stock.crnn_gru import CRNN_GRU

class CRNN_factory:
    def __init__(self,feature_num, filter_num, window, ticker_num, hidden_unit_num, hidden_layer_num, dropout):
        self.crnns = []
        self.crnns.append((CRNN_LSTM(feature_num, filter_num, window, ticker_num, hidden_unit_num, hidden_layer_num, dropout)))
        self.crnns.append((CRNN_RNN(feature_num, filter_num, window, ticker_num, hidden_unit_num, hidden_layer_num, dropout)))
        self.crnns.append((CRNN_GRU(feature_num, filter_num, window, ticker_num, hidden_unit_num, hidden_layer_num, dropout)))

    def get_model(self,code):
        for model in self.crnns:
            if model.get_code()==code:
                return model
        return None