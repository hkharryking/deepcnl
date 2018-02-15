class Learner_factory:
    def __init__(self):
        self.learners=[]
        self.learners.append(DeepCNL_learner())
        #self.learners.append(PCC_learner())
        #self.learners.append(DTW_learner())
        #self.learners.append(VWL_learner())

    def get_learner(self,code):
        for learner in self.learners:
            if learner.get_code()==code:
                return learner


class Graph_learner:
    def get_code(self):
        pass

class DeepCNL_learner(Graph_learner):
    def get_code(self):
        return 'DEEPCNL'
    def run(self,train_x, train_y,datatool,rare_ratio):
        print('[DeepCNL]')
        self.train_model(train_x, train_y)
        W=None
        for m in self.model.modules():
            if isinstance(m, nn.LSTM):
                (W_ii,W_if,W_ig,W_io)=m.weight_ih_l0.view(4,HIDDEN_UNIT_NUM,-1)  # LSTM weights, from input to the hidden layers
                W=W_ii+W_io+W_ig#+W_if#torch.mul(torch.mul(W_ii,W_io),W_ig)+W_if W_ii+
            if isinstance(m, nn.RNN):
                W=m.weight_ih_l0
            if isinstance(m, nn.GRU):
                (W_ir,W_iz, W_in)=m.weight_ih_l0.view(3,HIDDEN_UNIT_NUM,-1)
                W = W_iz+W_in+W_ir
        W=W.cumsum(dim=0)[HIDDEN_UNIT_NUM-1] #.cumsum(dim=0)
        W = W.sort(descending=True)
        E = W[1] # ticker dyadics
        W = W[0] # ticker weights
        g = nx.Graph()
        for k in range(0,int(TICKER_NUM*TICKER_NUM*rare_ratio)):# loaded edge numbers
            if W[k].cpu().data.numpy()[0] >0:
                (i,j)=datatool.check_dyadic(E[k].cpu().data.numpy()[0])
                i = datatool.check_ticker(i)
                j = datatool.check_ticker(j)
                g.add_edge(i,j)
        self.top_degree_nodes(g,TOP_DEGREE_NODE_NUM)

        #nx.draw(g, nx.spring_layout(g), with_labels=True,font_size=20)
        #plt.show()
        return g

class PCC_learner(Graph_learner):
    def get_code(self):
        return 'PCC'

    def run(self, train_x, train_y,datatool, rare_ratio):
        print('[Pearson correlation coefficients on time series tuples with (coefficient, p-value)]')
        g = nx.Graph()
        edgelist = {}
        for n in range(0, datatool.compare_data.shape[0]):
            t = datatool.compare_data[n]
            (i, j) = datatool.check_dyadic(n)
            i = datatool.check_ticker(i)
            j = datatool.check_ticker(j)
            prs = stats.pearsonr(t[0], t[1])
            if prs[1] <= 0.01 and prs[0] > 0:  # prs[1]: p value = edge_ratio; prs[0] = coefficient
                edgelist[(i, j)] = prs[0]
        edgelist = sorted(edgelist.items(), key=lambda x: x[1], reverse=True)
        for k in range(0, int(TICKER_NUM * TICKER_NUM * rare_ratio)):
            ((i, j), weight) = edgelist[k]
            g.add_edge(i, j)
        self.top_degree_nodes(g, TOP_DEGREE_NODE_NUM)
        return g

class DTW_learner(Graph_learner):
    def get_code(self):
        return 'DTW'

    def run(self, train_x, train_y, datatool, rare_ratio):
        print('[DTW graph finding]')
        g = nx.Graph()
        edgelist={}
        for n in range(0, datatool.compare_data.shape[0]):
            t = datatool.compare_data[n]
            (i, j) = datatool.check_dyadic(n)
            i = datatool.check_ticker(i)
            j = datatool.check_ticker(j)
            distance, path = fastdtw(t[0], t[1], dist=euclidean)
            if n%100==0:
                print(i,j,distance)
            edgelist[(i, j)] = 1.0/(distance+1)
        edgelist = sorted(edgelist.items(), key=lambda x: x[1], reverse=True)
        for k in range(0,int(TICKER_NUM*TICKER_NUM*rare_ratio)):
            ((i,j),weight)=edgelist[k]
            g.add_edge(i, j)
        self.top_degree_nodes(g, TOP_DEGREE_NODE_NUM)
        return g

class VWL_learner(Graph_learner):
    def get_code(self):
        return 'VWL'

    def run(self, train_x, train_y, datatool,rare_ratio):
        print('[Visibility graphs-WL kernel method]')
        g = nx.Graph()
        edgelist = {}
        for n in range(0, datatool.compare_data.shape[0]):
            t = datatool.compare_data[n]
            (i, j) = datatool.check_dyadic(n)
            i = datatool.check_ticker(i)
            j = datatool.check_ticker(j)
            g0 = visibility_graph(t[0])
            g1 = visibility_graph(t[1])
            weight = self.wlkernel.compare(g0, g1, h=10, node_label=False)
            print(i, j, weight)
            edgelist[(i, j)] = weight
        edgelist = sorted(edgelist.items(), key=lambda x: x[1], reverse=True)
        for k in range(0, int(TICKER_NUM * TICKER_NUM * rare_ratio)):
            ((i, j), weight) = edgelist[k]
            g.add_edge(i, j)
        self.top_degree_nodes(g, TOP_DEGREE_NODE_NUM)
        return g