# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 14:41:42 2017

@author: wangyue
Stock price preidiction based on a mixed deep learning model
with crossentropy, mse and differential reward
correct the code with:
Convolution with windowed dyadic timeseries input, LSTM with n*(n-1)/2 edges input
Mimic the market index with same rise and fall pattern in order to find out the relationship between assets

update on Fri Dec 20 2017
TODO: test the GRU
update the cpu test version to mask the bug
completed pearson and visibility wl kernel methods
tested MSE loss < Cross Entropy loss
updated LSTM network finder
fixed bugs at the hidden status initialing process
tested different optimizer as adam, adagrad, adadelta, rmsprop
tested optimize by SGD with momentum
updated custom regularizer
updated xavier,orthogonal initialization,https://gist.github.com/kaniblu/81828dfcf5cca60ae93f4d7bd19aeac5

update on Fri Dec 20 2017
TODO: test the GRU
completed pearson and visibility wl kernel methods
tested MSE loss < Cross Entropy loss
updated LSTM network finder
fixed bugs at the hidden status initialing process
tested different optimizer as adam, adagrad, adadelta, rmsprop
tested optimize by SGD with momentum
updated custom regularizer
updated xavier,orthogonal initialization,https://gist.github.com/kaniblu/81828dfcf5cca60ae93f4d7bd19aeac5

update on Fri Dec 19 2017
TODO: test the MSE loss and RNN or GRU
tested MSE loss < Cross Entropy loss
updated LSTM network finder
fixed bugs at the hidden status initialing process
tested different optimizer as adam, adagrad, adadelta, rmsprop
tested optimize by SGD with momentum
updated custom regularizer
updated xavier,orthogonal initialization,https://gist.github.com/kaniblu/81828dfcf5cca60ae93f4d7bd19aeac5

update on Fri Dec 18 2017
fixed bugs at the hidden status initialing process
tested different optimizer as adam, adagrad, adadelta, rmsprop
tested optimize by SGD with momentum
updated custom regularizer
TODO: try orthogonal initialization,https://gist.github.com/kaniblu/81828dfcf5cca60ae93f4d7bd19aeac5

update on Fri Dec 13 2017
fixed bugs at the hidden status initialing process
TODO： test different optimizer as adam, adagrad, adadelta, rmsprop
TODO: or optimize by SGD with momentum
TODO: try orthogonal initialization,https://gist.github.com/kaniblu/81828dfcf5cca60ae93f4d7bd19aeac5


update on Fri Dec 10 2017
complete the batch normalization
TODO： test different optimizer as adam, adagrad, adadelta, rmsprop
TODO: or optimize by SGD with momentum

update on Fri Dec 9 2017
TODO: batch normalization
TODO： test different optimizer as adam, adagrad, adadelta, rmsprop
TODO: or optimize by SGD with momentum

update on Fri Dec 8 2017
update content: cuda version

update on Sun Nov 26 2017
update content: preprocess to align all the time series data in the S&P 500
"""
import torch
from torch import nn
from torch.autograd import Variable
from torch.backends import cudnn
import networkx as nx
from crnn_stock.crnn_factory import CRNN_factory
from crnn_stock.data_util import Data_util
import scipy.stats as stats
from visibility_graph import visibility_graph
from crnn_stock.wlkernel import WLkernerl
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import time
import itertools

# Hyper Parameters
LEARNER_CODE = 'DEEPCNL'  #DEEPCNL, PCC, DTW, VWL
CRNN_CODE = 'CRNN_LSTM' # CRNN_RNN,CRNN_LSTM,CRNN_GRU
DNL_INPLEMENTATION = ['igo','g','io','igof']
WINDOW = 32      # WINDOW size for the time series 32 64
FEATURE_NUM = 4     # feature (e.g. open high, low, close,volume) number, drop open
FILTERS_NUM = 16  # CNN kernel number, LSTM Linear layer merged feature number
LR = 0.0005           # learning rate 0.001 for optimizer
EPOCH_NUM=200       # Iteration times for training data
TICKER_NUM = 470     # S&P500  maximum in data set for 470 tickers for 400 more will be the cudnn bug on batchnorm1d
YEAR_SEED = 2 # train_period = 2010+seed-1-1 to 2010+seed-12-31; test_period = 2011+seed-1-1 to 2011+seed-6-30
HIDDEN_UNIT_NUM = 256
HIDDEN_LAYER_NUM = 2
LAM=0.0005
DROPOUT=0.35
DATA_PATH = ".../data/kaggle/prices-split-adjusted.csv"
SPY_PATH = '.../data/SPY20000101_20171111.csv'
SP500_PATH = '.../data/SP500^GSPC20000101_20171111.csv'
RARE_RATIO = 0.002 # 0.001 for 470 / 0.01 for less
TOP_DEGREE_NODE_NUM=20

# 50
# https://www.guggenheiminvestments.com/etf/fund/xlg-guggenheim-sp-500-top-50-etf/holdings
OEX = ['AAPL','MSFT','AMZN','FB','BRKB','JNJ','JPM','XOM','GOOG','GOOGL','BAC','WFC','CVX','HD','PG','UNH','T','PFE','V','VZ','INTC','C','CSCO','BA','CMCSA','KO','DWDP','MRK','PEP','DIS','ABBV','PM','ORCL','MA','GE','WMT','MMM','IBM','MCD','AMGN','MO','HON','TXN','MDT','UNP','SLB','GILD','ABT','BMY','QCOM','CAT','UTX','ACN','PCLN','PYPL','UPS','GS','USB','SBUX','LOW','COST','NKE','LMT','LLY','CVS','CELG','MS','BIIB','AXP','TWX','COP','NEE','BLK','CHTR','CL','FDX','WBA','MDLZ','DHR','BK','AGN','OXY','GD','RTN','GM','MET','AIG','DUK','MON','SPG','COF','KHC','F','EMR','HAL','SO','TGT','FOXA','KMI','EXC','ALL','BLKFDS','FOX','USD','MSFUT','ESH8']

# 100
# https://www.ishares.com/us/products/239723/ishares-sp-100-etf
XLG = ['AAPL','MSFT','AMZN','FB','BRK','B','JNJ','JPM','XOM','GOOG','GOOGL','BAC','WFC','CVX','HD','PG','UNH','T','PFE','V','VZ','INTC','C','CSCO','BA','CMCSA','KO','MRK','PEP','DIS','ABBV','PM','ORCL','MA','GE','WMT','MMM','IBM','MCD','AMGN','MO','HON','MDT','UNP','AVGO','SLB','GILD','BMY','UTX','PCLN','SBUX','CELG','X9USDFUTC']

# 200
# https://www.ishares.com/us/products/239721/ishares-russell-top-200-etf
IWL = ['AAPL','MSFT','AMZN','FB','BRKB','JNJ','JPM','XOM','GOOG','GOOGL','BAC','WFC','CVX','HD','PG','UNH','T','V','PFE','VZ','INTC','C','CSCO','BA','CMCSA','KO','DWDP','DIS','PEP','MRK','ABBV','PM','MA','GE','WMT','ORCL','MMM','IBM','MCD','AMGN','MO','NVDA','HON','TXN','MDT','UNP','AVGO','SLB','GILD','BMY','QCOM','UTX','ABT','ACN','ADBE','CAT','PCLN','PYPL','UPS','GS','NFLX','USB','SBUX','LOW','TMO','LLY','COST','LMT','NKE','CVS','CELG','PNC','CRM','BIIB','AXP','COP','BLK','MS','TWX','NEE','CB','FDX','WBA','SCHW','CHTR','CL','EOG','MDLZ','ANTM','AMAT','DHR','BDX','AGN','AET','OXY','AMT','BK','RTN','GM','AIG','DUK','ADP','SYK','GD','DE','PRU','CI','MON','ITW','SPG','ATVI','CME','NOC','COF','TJX','CSX','MU','D','ISRG','KHC','MET','F','EMR','PX','PSX','TSLA','ESRX','HAL','SPGI','SO','NSC','CTSH','ICE','MAR','VLO','CCI','TGT','BBT','MMC','KMB','NXPI','INTU','HPQ','DAL','STT','HUM','VRTX','FOXA','WM','ALL','LYB','TRV','KMI','EXC','ETN','EBAY','BSX','JCI','MCK','LUV','APD','STZ','ECL','SHW','EQIX','AFL','AON','BAX','GIS','EA','AEP','APC','PXD','SYY','GLW','REGN','PPG','PSA','EL','CCL','YUM','MNST','HPE','ALXN','BLKFDS','LVS','HCA','KR','ADM','PCG','EQR','CBS','TMUS','USD','FOX','BEN','DISH','VMW','BHF','S','JPFFT','ESH8']

    
class Experimental_platform:
    def __init__(self,datatool):
        self.model=None
        self.datatool=datatool
        self.wlkernel=WLkernerl()
        self.crnn_factory=CRNN_factory(FEATURE_NUM, FILTERS_NUM, WINDOW, TICKER_NUM, HIDDEN_UNIT_NUM, HIDDEN_LAYER_NUM, DROPOUT)

    def regularizer(self,lam,loss):
        li_reg_loss = 0
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                temp_loss = torch.sqrt(torch.sum(torch.pow(m.weight,2)))
                li_reg_loss += temp_loss
        loss = loss + lam*li_reg_loss
        return loss

    
    def train_model(self,x,y):
        self.model=None
        self.model = self.crnn_factory.get_model(CRNN_CODE)
        self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)   # optimize
        #self.optimizer = torch.optim.Adadelta(self.model.parameters())
        #self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=LR)
        #self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=LR,alpha=0.9)
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.15, momentum=0.8)
        self.loss_func = torch.nn.CrossEntropyLoss()#size_average=True
        #self.loss_func = nn.MSELoss()
        #loss_func = Reward_loss()

        # train the deep learning model
        print('start training the deep learning model')
        cudnn.benchmark = True
        inputs = Variable(torch.from_numpy(x)).float().cuda()   # shape (batch, time_step, input_size)
        targets = Variable(torch.from_numpy(y)).long().cuda()
        self.model.train()
        prediction=None
        for epoch in range(EPOCH_NUM):
            self.model.zero_grad()
            self.model.hidden = self.model.init_hidden()

            prediction = self.model(inputs)   # model output
            #convert the prediction into a classification series
            loss = self.loss_func.forward(self.model.classify_result(prediction), targets)         # CrossEntropy loss
            #loss = self.loss_func.forward(prediction, targets.float()) # MSE loss
            loss = self.regularizer(LAM,loss)
            self.optimizer.zero_grad()                   # clear gradients for this training step
            loss.backward()                         # backpropagation, compute gradients
            self.optimizer.step()                        # apply gradients
            if epoch%10==0:
                print('epoch',epoch,loss.data.cpu().numpy()[0])
        result=self.model.classify_result(prediction)
        #pred = result.max(1, keepdim=True)[1]
        torch.save(self.model, 'model.pt')
        return loss.data.cpu().numpy()[0] # return train loss
    
    def test_model(self,x,y):
        #torch.backends.cudnn.benchmark = True
        self.model=torch.load('model.pt', map_location=lambda storage, loc: storage) # load data into cpu and test
        self.model.eval()  # TODO: weird bug for no-contigous input at the batch normal
        self.loss_func=self.loss_func.cpu()
        print('testing the learned model')
        correct=0
        inputs = Variable(torch.from_numpy(x)).cpu()
        targets = Variable(torch.from_numpy(y)).cpu()
        #inputs=inputs.contiguous()

        prediction = self.model(inputs.float())   # test output
        result=self.model.classify_result(prediction)
        #convert the prediction into a classification series
        loss = self.loss_func.forward(result, targets.long())
        print('test loss',loss.cpu().data.numpy())
        pred = result.max(1, keepdim=True)[1]
        correct = pred.eq(targets.view_as(pred).long()).sum()
        acc=(correct.float()/len(targets)).cpu().data.numpy()
        print('accuracy:',acc)
        return loss.cpu().data.numpy()[0],acc # return test loss and accuracy

    def top_degree_nodes(self,g):
        result=sorted(g.degree,key= lambda x: x[1],reverse=True)
        node_num=len(g.nodes())
        N=TOP_DEGREE_NODE_NUM
        if node_num<TOP_DEGREE_NODE_NUM:
            N=node_num
        for n in range(0,N):
            print(result[n])

    def DNL_graph_learning(self,dnl_implementation,rare_ratio):
        W = None
        for m in self.model.modules():
            if isinstance(m, nn.LSTM):
                (W_ii, W_if, W_ig, W_io) = m.weight_ih_l0.view(4, HIDDEN_UNIT_NUM,
                                                               -1)  # LSTM weights, from input to the hidden layers
                if dnl_implementation == 'igo':
                    W = W_ii + W_ig + W_io  # paper version
                if dnl_implementation == 'igof':
                    W = W_ii + W_io + W_ig + W_if  # full version
                if dnl_implementation == 'io':
                    W = W_ii + W_io  # input and output gates
                if dnl_implementation == 'g':
                    W = W_ig  # input and output gates

            if isinstance(m, nn.RNN):
                W = m.weight_ih_l0
            if isinstance(m, nn.GRU):
                (W_ir, W_iz, W_in) = m.weight_ih_l0.view(3, HIDDEN_UNIT_NUM, -1)
                W = W_iz + W_in + W_ir
        W = W.cumsum(dim=0)[HIDDEN_UNIT_NUM - 1]  # .cumsum(dim=0)
        W = W.sort(descending=True)
        E = W[1]  # ticker dyadics
        W = W[0]  # ticker weights
        g = nx.Graph()
        edge_bunch = []
        for k in range(0, int(TICKER_NUM * (TICKER_NUM - 1) * 0.5 * rare_ratio)):  # loaded edge numbers
            if W[k].cpu().data.numpy()[0] > 0:
                (i, j) = self.datatool.check_dyadic(E[k].cpu().data.numpy()[0])
                i = self.datatool.check_ticker(i)
                j = self.datatool.check_ticker(j)
                # g.add_edge(i,j)
                edge_bunch.append((i, j, W[k].cpu().data.numpy()[0]))
        g.add_weighted_edges_from(edge_bunch)
        self.top_degree_nodes(g)

        # nx.draw(g, nx.spring_layout(g), with_labels=True,font_size=20)
        # plt.show()
        return g


    def deep_CNL(self, dnl_implementation, train_x, train_y, rare_ratio):
        print('[DeepCNL]')
        self.train_model(train_x, train_y)
        return self.DNL_graph_learning(dnl_implementation, rare_ratio)

    def Pearson_cor(self,rare_ratio):
        print('[Pearson correlation coefficients on time series tuples with (coefficient, p-value)]')
        g = nx.Graph()
        edgelist = {}
        for n in range(0,self.datatool.compare_data.shape[0]):
            t=datatool.compare_data[n]
            (i, j) = self.datatool.check_dyadic(n)
            i = self.datatool.check_ticker(i)
            j = self.datatool.check_ticker(j)
            prs=stats.pearsonr(t[0],t[1])
            if prs[1] <= 0.01 and prs[0]>0: # prs[1]: p value = edge_ratio; prs[0] = coefficient
                edgelist[(i,j)]=prs[0]
        edgelist=sorted(edgelist.items(),key=lambda x:x[1],reverse=True)
        edge_bunch = []
        for k in range(0, int(TICKER_NUM * (TICKER_NUM - 1) * 0.5 * rare_ratio)):
            ((i, j), weight) = edgelist[k]
            # g.add_edge(i, j)
            edge_bunch.append((i, j, weight))
        g.add_weighted_edges_from(edge_bunch)
        self.top_degree_nodes(g)
        return g


    def VWL_graph(self,rare_ratio):
        print('[Visibility graphs-WL kernel method]')
        g=nx.Graph()
        edgelist={}
        for n in range(0,self.datatool.compare_data.shape[0]):
            t=datatool.compare_data[n]
            (i, j) = self.datatool.check_dyadic(n)
            i = self.datatool.check_ticker(i)
            j = self.datatool.check_ticker(j)
            g0=visibility_graph(t[0])
            g1=visibility_graph(t[1])
            weight=self.wlkernel.compare(g0,g1,h=10,node_label=False)
            print(i,j,weight)
            edgelist[(i, j)] = weight
        edgelist = sorted(edgelist.items(), key=lambda x: x[1], reverse=True)
        edge_bunch = []
        for k in range(0, int(TICKER_NUM * (TICKER_NUM - 1) * 0.5 * rare_ratio)):
            ((i, j), weight) = edgelist[k]
            # g.add_edge(i, j)
            edge_bunch.append((i, j, weight))
        g.add_weighted_edges_from(edge_bunch)
        self.top_degree_nodes(g, TOP_DEGREE_NODE_NUM)
        return g

    def DTW_graph(self,rare_ratio):
        print('[DTW graph finding]')
        g = nx.Graph()
        edgelist={}
        for n in range(0, self.datatool.compare_data.shape[0]):
            t = datatool.compare_data[n]
            (i, j) = self.datatool.check_dyadic(n)
            i = self.datatool.check_ticker(i)
            j = self.datatool.check_ticker(j)
            distance, path = fastdtw(t[0], t[1], dist=euclidean)
            if n%100==0:
                print(i,j,distance)
            edgelist[(i, j)] = 1.0/(distance+1)
        edgelist = sorted(edgelist.items(), key=lambda x: x[1], reverse=True)
        edge_bunch=[]
        for k in range(0,int(TICKER_NUM*(TICKER_NUM-1)*0.5*rare_ratio)):
            ((i,j),weight)=edgelist[k]
            #g.add_edge(i, j)
            edge_bunch.append((i,j,weight))
        g.add_weighted_edges_from(edge_bunch)
        self.top_degree_nodes(g, TOP_DEGREE_NODE_NUM)
        return g


    '''
    Experiments
    '''

    def coverage_comparison(self):
        benchmark=[['NFLX','FFIV','CMI','AIG','ZION','HBAN','AKAM','PCLN','WFMI','Q'], # 2010
                   ['COG','EP','ISRG','MA','BIIB','HUM','CMG','PRGO','OKS','ROST'], # 2011
                   ['HW','DDD','REGN','LL','PHM','MHO','AHS','VAC','S','EXH'], # 2012
                   ['NFLX','MU','BBY','DAL','CELG','BSX','GILD','YHOO','HPQ','LNC'], # 2013
                   ['LUV','EA','EW','AGN','MNK','AVGO','GMCR','DAL','RCL','MNST'], # 2014
                   ['NFLX','AMZN','ATVI','NVDA','CVC','HRL','VRSN','RAI','SBUX','FSLR'], # 2015
                   ['NVDA','OKE','FCX','CSC','AMAT','PWR','NEM','SE','BBY','CMI']] # 2016
        for seed in [2]:
            train_period, test_period = self.period_generator(seed)
            train_x = self.datatool.load_x(train_period)  # TRAIN_PERIOD
            train_y = self.datatool.load_y(train_period)  # TRAIN_PERIOD
            if LEARNER_CODE == 'DEEPCNL':
                g = self.deep_CNL('igo',train_x, train_y, RARE_RATIO)
            if LEARNER_CODE == 'PCC':
                g = self.Pearson_cor(RARE_RATIO)
            sum_count = 0.
            #sorted_degrees = sorted(g.degree, key=lambda x: x[1], reverse=True) # thus the higher the rank the better.
            for ticker in benchmark[seed]:
                if g.degree(ticker)!=None and isinstance(g.degree(ticker),int)==1:
                    print('[COVERED]', ticker)
                    sum_count+=1
            print(str(2010 + seed),'covered number:',sum_count)

    def get_rank(self,sorted_degree_list,ticker):
        for i in range(0,len(sorted_degree_list)):
            if ticker == sorted_degree_list[i][0]:
                return i



    def rise_fall_prediction(self, seed):
        train_period, test_period = self.period_generator(seed)
        print('[RISE-FALL PREDICT TASK]')
        print('train with data in', train_period)
        print('test with data in', test_period)
        train_x = self.datatool.load_x(train_period)
        train_y = self.datatool.load_y(train_period)
        test_x = self.datatool.load_x(test_period)
        test_y = self.datatool.load_y(test_period)
        train_loss = 0.
        test_loss = 0.
        accuracy = 0.
        print('[REPEAT Experiments, iteration]',seed)
        train_loss=self.train_model(train_x,train_y)
        loss,acc=self.test_model(test_x,test_y)
        test_loss=loss
        accuracy=acc
        print('[**RESULT**]')
        print('Train loss',train_loss)
        print('Test loss', test_loss)
        print('accuracy', accuracy)

    def DNL_density_comparison(self,seed):
        print('DNL implemented with gates:')
        train_period,test_period = self.period_generator(seed)
        print('train with data in',train_period)
        # load data
        train_x = self.datatool.load_x(train_period) #TRAIN_PERIOD
        train_y = self.datatool.load_y(train_period) # TRAIN_PERIOD
        self.train_model(train_x, train_y)
        for dnl in DNL_INPLEMENTATION:
            print('DeepCNL-'+dnl)
            g=self.DNL_graph_learning(dnl,RARE_RATIO)
            print('edge density')
            print('[XLG 50]', experiment.edge_density(g, XLG))
            print('[OEX 100]', experiment.edge_density(g, OEX))
            print('[IWL 200]', experiment.edge_density(g, IWL))

    def ALL_density_comparison(self, seed):
        print('general comparison with DeepCNL-igo, PCC')
        g = experiment.influential_asset_finding(seed)
        print('average weight')
        print('edge density')
        print('[XLG 50]', experiment.edge_density(g, XLG))
        print('[OEX 100]', experiment.edge_density(g, OEX))
        print('[IWL 200]', experiment.edge_density(g, IWL))

    def correlation_degree_comparison(self, seed):
        g = experiment.influential_asset_finding(seed)
        print('average weight')
        print('[XLG]', experiment.average_weight(nx.subgraph(g, XLG)))
        print('[OEX]', experiment.average_weight(nx.subgraph(g, OEX)))
        print('[IWL]', experiment.average_weight(nx.subgraph(g, IWL)))
        print('[SPY]', experiment.average_weight(g))

    '''
    seed from 0 to 6
    '''
    def period_generator(self,seed):
        train_period = [str(2010 + seed) + '-1-1', str(2010 + seed) + '-12-31']
        test_period = [str(2011 + seed) + '-1-1', str(2011 + seed) + '-6-30']
        return train_period,test_period

    def influential_asset_finding(self,seed):
        print('DNL implemented with gates:')
        train_period,test_period = self.period_generator(seed)
        print('train with data in',train_period)
        # load data
        train_x = self.datatool.load_x(train_period) #TRAIN_PERIOD
        train_y = self.datatool.load_y(train_period) # TRAIN_PERIOD

        if LEARNER_CODE == 'DEEPCNL':
            g = self.deep_CNL(train_x,train_y,RARE_RATIO)
        if LEARNER_CODE == 'PCC':
            g = self.Pearson_cor(RARE_RATIO)
        if LEARNER_CODE == 'VWL':
            g = self.VWL_graph(RARE_RATIO)
        if LEARNER_CODE == 'DTW':
            g = self.DTW_graph(RARE_RATIO)

        #nx.draw(g, nx.spring_layout(g), with_labels=True, font_size=20)
        #plt.show()
        return g

    def average_weight(self,g):
        ws = nx.get_edge_attributes(g, 'weight')
        result=0.
        for e in ws:
            result+=ws[e]
        return result/len(ws)


    def edge_density(self,g,etf):
        c=0.
        combines=[c for c in itertools.combinations(range(len(etf)),2)]
        for (i,j) in combines:
            if g.has_edge(etf[i],etf[j]):
                c+=1
        return c/len(combines)







if __name__ == "__main__":
    print('[Parameters]***********************************************')
    print('LEANER_CODE',LEARNER_CODE)
    print('CRNN_CODE', CRNN_CODE)
    print('DNL_IMPLEMENTATIONS', DNL_INPLEMENTATION)
    print('[WINDOW]',WINDOW)
    print('[FEATURE_NUM]',FEATURE_NUM)
    print('[FILTERS_NUM]',FILTERS_NUM)
    print('[LR]',LR)
    print('[DeepCNL epoches]',EPOCH_NUM)
    print('[TICKER_NUM]',TICKER_NUM)
    print('[HIDDEN_UNIT_NUM]',HIDDEN_UNIT_NUM)
    print('[HIDDEN_LAYER_NUM]',HIDDEN_LAYER_NUM)
    print('[LAM]',LAM)
    print('[DROPOUT]',DROPOUT)
    print('[RARE_RATIO]',RARE_RATIO)
    print('[TOP_DEGREE_NODE_NUM]',TOP_DEGREE_NODE_NUM)
    print('[Parameters]***********************************************')
    datatool=Data_util(TICKER_NUM,WINDOW,FEATURE_NUM,DATA_PATH,SPY_PATH)
    experiment=Experimental_platform(datatool)
    start = time.clock()
    experiment.coverage_comparison()
    #for seed in [5,6]:
    #experiment.DNL_density_comparison(0)
        #experiment.ALL_density_comparison(seed)
        #   experiment.rise_fall_prediction(seed)

    #experiment.correlation_degree_comparison(YEAR_SEED)
    #experiment.sum_degree()


    elapsed = (time.clock() - start)
    print("Time used:", elapsed, 'Seconds')
    
    
   
    
