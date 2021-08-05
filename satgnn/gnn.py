import torch
from torch import nn
import dgl.function as fn


VAR_FEATS = 2 
U_FEATS = 4 

class MessageGNN(nn.Module):
    def __init__(self,emb_dim, c_feats, v_feats, u_feats, e_feats):
        super(MessageGNN, self).__init__()
        
        self.ebm_dim = emb_dim
        
        self.message_cv_layer = nn.Sequential(nn.Linear(e_feats+emb_dim, emb_dim), nn.LeakyReLU(0.1))
        self.message_vc_layer = nn.Sequential(nn.Linear(e_feats+emb_dim, emb_dim), nn.LeakyReLU(0.1))    
        
        self.c_update_layer = nn.Sequential(nn.Linear(c_feats+3*emb_dim, emb_dim), nn.LeakyReLU(0.1))
        self.v_update_layer = nn.Sequential(nn.Linear(v_feats+3*emb_dim, emb_dim), nn.LeakyReLU(0.1))
        self.u_update_layer = nn.Sequential(nn.Linear(u_feats+3*emb_dim, emb_dim), nn.LeakyReLU(0.1))
        
    
    def message_vc_func(self, edges):
        edge_features = edges.data['edge_sat']
        src_embeddings = edges.src['embeddings']

        # print(edge_features.shape)
        # print(src_embeddings.shape)
        # print(torch.cat((edge_features, src_embeddings), 1).shape)
        message = self.message_vc_layer(torch.cat((edge_features, src_embeddings), 1))

        return {'m' : message}
    
    
    def message_cv_func(self, edges):
        edge_features = edges.data['edge_sat']
        src_embeddings = edges.src['embeddings']

        message = self.message_cv_layer(torch.cat((edge_features, src_embeddings), 1))

        return {'m' : message}
    
    
    def message_u_func(self, edges):
        return {'m' : edges.src['embeddings']}
    
    
    def update_clause_nodes(self, nodes):
        message = nodes.data['h']
        context = nodes.data['ctx']
        current_node = nodes.data['embeddings']
        clause_feats = nodes.data['features']

        updated = self.c_update_layer(torch.cat([clause_feats,message,context,current_node],1)) 

        return {'embeddings' : updated}

    
    def update_variable_nodes(self, nodes):
        message = nodes.data['h']
        context = nodes.data['ctx']
        current_node = nodes.data['embeddings']
        var_feats = nodes.data['features']

        updated = self.v_update_layer(torch.cat([var_feats,message,context,current_node],1)) 

        return {'embeddings' : updated}
    
    
    def update_context(self, graph):
        clause_embeddings = graph.nodes['context'].data['c']
        variable_embeddings = graph.nodes['context'].data['v']
        current_node = graph.nodes['context'].data['embeddings']
        ctx_feats = graph.nodes['context'].data['features']
        
        graph.nodes['context'].data['embeddings'] = self.u_update_layer(torch.cat([ctx_feats,clause_embeddings,variable_embeddings,current_node],1))


    
    def forward(self, graph):
        graph.multi_update_all({
                    ('context', 'has_node', 'variable'): (self.message_u_func, fn.mean('m', 'ctx')),
                    ('context', 'has_node', 'clause'): (self.message_u_func, fn.mean('m', 'ctx')),},
                   "mean")
           
        graph.multi_update_all({
                    'assigns': (self.message_vc_func, fn.mean('m', 'h'), self.update_clause_nodes),
                    'contains': (self.message_cv_func, fn.mean('m', 'h'), self.update_variable_nodes),},
                   "mean")
        
        graph.multi_update_all({
                    ('clause', 'has_context', 'context'): (self.message_u_func, fn.mean('m', 'c')),
                    ('variable', 'has_context', 'context'): (self.message_u_func, fn.mean('m', 'v')),},
                   "mean")

        self.update_context(graph)
                
        return graph


class SatGNN(nn.Module):
    def __init__(self, emb_dim, num_var, var_feats, u_feats, seq_length=5, hidden=2, hidden_dim=128):
        super(SatGNN, self).__init__()
        
        self.num_var = num_var
        
        self.clause_feats = 2
        self.var_feats = var_feats
        self.u_feats = u_feats
        self.e_feats = 2
    
        self.u_emb = nn.Linear(self.u_feats, emb_dim)
        self.v_emb = nn.Parameter(torch.rand(emb_dim)) 
        self.c_emb = nn.Linear(self.clause_feats, emb_dim)
            
        self.message_layers = nn.Sequential(*nn.ModuleList([MessageGNN(emb_dim, self.clause_feats, self.var_feats, self.u_feats, e_feats=self.e_feats) 
                                             for _ in range(seq_length)])) 
        
        if hidden > 0:
            module_list = nn.ModuleList([nn.Linear(emb_dim,hidden_dim), nn.ReLU()])
            for i in range(hidden):            
                module_list.append(nn.Linear(hidden_dim,hidden_dim))
                module_list.append(nn.ReLU())
            module_list.append(nn.Linear(hidden_dim, 1))
            self.Q_layer = nn.Sequential(*module_list)  
        else:
            self.Q_layer = nn.Linear(emb_dim, 1)
        
        
    def forward(self, graph):
        # initial embeddings
        graph.nodes['variable'].data['embeddings'] = self.v_emb.repeat(graph.number_of_nodes('variable'),1)
        graph.nodes['clause'].data['embeddings'] = self.c_emb(graph.nodes['clause'].data['features'])
        graph.nodes['context'].data['embeddings'] = self.u_emb(graph.nodes['context'].data['features'])
        

        # message passing
        self.message_layers(graph)

        num_var = int(graph.number_of_nodes('variable') / graph.number_of_nodes('context'))
        
        # fully connected
        q_values = self.Q_layer(graph.nodes['variable'].data['embeddings']).reshape(-1,num_var)
        
        return q_values




class SatGNNRecursive(nn.Module):
    def __init__(self, emb_dim, num_var, var_feats, u_feats, seq_length=5, hidden=2, hidden_dim=128):
        super(SatGNNRecursive, self).__init__()
        
        self.num_var = num_var
        
        self.clause_feats = 2
        self.var_feats = var_feats
        self.u_feats = u_feats
        self.e_feats = 2
    
        self.u_emb = nn.Linear(self.u_feats, emb_dim)
        self.v_emb = nn.Linear(self.var_feats, emb_dim)
        # self.v_emb = nn.Parameter(torch.rand(emb_dim)) 
        self.c_emb = nn.Linear(self.clause_feats, emb_dim)
            
        self.message_layer = MessageGNN(emb_dim, self.clause_feats, self.var_feats, self.u_feats, e_feats=self.e_feats)
        self.seq_length = seq_length
        
        if hidden > 0:
            module_list = nn.ModuleList([nn.Linear(emb_dim,hidden_dim), nn.ReLU()])
            for i in range(hidden):            
                module_list.append(nn.Linear(hidden_dim,hidden_dim))
                module_list.append(nn.ReLU())
            module_list.append(nn.Linear(hidden_dim, 1))
            self.Q_layer = nn.Sequential(*module_list)  
        else:
            self.Q_layer = nn.Linear(emb_dim, 1)
        
        
    def forward(self, graph):
        # initial embeddings
        # graph.nodes['variable'].data['embeddings'] = self.v_emb.repeat(graph.number_of_nodes('variable'),1)
        graph.nodes['variable'].data['embeddings'] = self.v_emb(graph.nodes['variable'].data['features'])
        graph.nodes['clause'].data['embeddings'] = self.c_emb(graph.nodes['clause'].data['features'])
        graph.nodes['context'].data['embeddings'] = self.u_emb(graph.nodes['context'].data['features'])
        

        # message passing
        for i in range(self.seq_length):
            graph = self.message_layer(graph)

        
        num_var = int(graph.number_of_nodes('variable') / graph.number_of_nodes('context'))
        
        # fully connected
        q_values = self.Q_layer(graph.nodes['variable'].data['embeddings']).reshape(-1,num_var)
        
        return q_values
