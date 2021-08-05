import torch
import dgl
import torch.nn.functional as F
import dgl.function as fn

class SatEnv():
    def __init__(self, adjacency, args):
        self.args = args
        self.num_var = adjacency.shape[0]
        self.num_clause = adjacency.shape[1]
        self.backtrack = False
        
        edge_index = (adjacency != 0).nonzero().T
                
        graph_data = {('variable', 'assigns', 'clause'): (edge_index[0], edge_index[1]),
                       ('clause', 'contains', 'variable'): (edge_index[1], edge_index[0]),
                       ('variable', 'has_context', 'context'): (torch.arange(self.num_var, dtype=torch.int64), torch.zeros(self.num_var, dtype=torch.int64)),
                       ('clause', 'has_context', 'context'): (torch.arange(self.num_clause, dtype=torch.int64), torch.zeros(self.num_clause, dtype=torch.int64)),
                       ('context', 'has_node', 'clause'): (torch.zeros(self.num_clause, dtype=torch.int64), torch.arange(self.num_clause, dtype=torch.int64)),
                       ('context', 'has_node', 'variable'): (torch.zeros(self.num_var, dtype=torch.int64), torch.arange(self.num_var, dtype=torch.int64)),}
        
        self.g = dgl.heterograph(graph_data)
        self.state = None
        
        ## add edge weight as one-hot
        self.g.edges['contains'].data['value'] = F.relu(torch.tensor([1,-1]).reshape(2,1) * adjacency[tuple(edge_index)]).T
        self.g.edges['assigns'].data['value'] = F.relu(torch.tensor([1,-1]).reshape(2,1) * adjacency[tuple(edge_index)]).T

        self.var_degrees = torch.tensor([self.g.in_degree(i,etype='contains') for i in range(self.num_var)])



        
    
    def propagate_clause_values(self, state):
        var_assignment = state['assignment']
        self.g.nodes['variable'].data['value'] =  torch.stack([var_assignment, (1-var_assignment)]).T
        self.g.apply_nodes(lambda node: {'stack': torch.stack([node.data['value'],(1-node.data['value'])], dim=1)}, ntype='variable')
        
        ## assigns edge features
        self.g.apply_edges(fn.u_dot_e('stack', 'value', 'edge_sat'), etype='assigns')
        self.g.apply_edges(fn.v_dot_e('stack', 'value', 'edge_sat'), etype='contains')


        self.g.edges['assigns'].data['edge_sat'] = self.g.edges['assigns'].data['edge_sat'].reshape(-1,2)
        self.g.edges['contains'].data['edge_sat'] = self.g.edges['assigns'].data['edge_sat'].reshape(-1,2)
                
        # clause node features
        self.g.multi_update_all({
                    ('variable', 'assigns', 'clause'): (lambda edges: {'m': edges.data['edge_sat'][:,0]}, fn.max('m', 'value'))},
                   "max")
        

        self.g.nodes['clause'].data['value'] = torch.stack((self.g.nodes['clause'].data['value'],(1-self.g.nodes['clause'].data['value'])),1)

        self.g.multi_update_all({
                    ('clause', 'contains', 'variable'): (self.message_temp_clause_vals, fn.mean('m', 'sat_value'))}, 'mean')

        # print(self.g.nodes['variable'].data['sat_sum'])
        # print(torch.sum(self.g.nodes['variable'].data['sat_sum']))
        # error
        # self.g.nodes['variable'].data['sat_value'] = torch.div(self.g.nodes['variable'].data['sat_sum'], self.var_degrees)
        


        
        # self.g.multi_update_all({
        #             ('variable', 'has_context', 'context'): (self.message_temp_clause_vals, fn.mean('val', 'val2'))})
    

    def message_temp_clause_vals(self, edges):
        # print(edges.src['value'])
        # error
        return {'m' : edges.src['value'][:,0]}


#         ## TEMP!! 
#         self.g.multi_update_all({
#                     ('clause', 'contains', 'variable'): (self.message_temp_clause_vals, fn.mean('val', 'val2'))}
            
#         self.g.multi_update_all({
#                     ('variable', 'has_context', 'context'): (self.message_temp_clause_vals, fn.mean('val', 'val2')),
    
#     def message_temp_clause_vals(self, edges):
#         return {'val' : edges.src['value'][0]}
            
#     def message_temp_clause_vals(self, edges):
#         return {'val' : edges.src['value'][0]}
            
    
    
    def get_sat_value(self):
        if self.args.reward_type == 'clause':
            clause_values = self.g.nodes['clause'].data['value'][:,0]
            sat_value = torch.sum(clause_values)/clause_values.shape[0]
        elif self.args.reward_type == 'variable':
            sat_value = torch.sum(self.g.nodes['variable'].data['sat_value'])/self.num_var
        return sat_value



    
    def set_sat_values(self, state):
        state['current_sat_value'] = self.get_sat_value()
        state['is_increasing'] = state['current_sat_value'] > state['best_sat_value']
        
        if state['is_increasing']:
            state['best_sat_value'] = state['current_sat_value']
            state['best_assignment'] = state['assignment']
        
        state['difference_to_best_observed'] = state['best_sat_value'] - state['current_sat_value']
        state['distance_to_best'] = self.num_var - torch.sum(torch.eq(state['assignment'], state['best_assignment'])).float()
        
        return state
      
    
    def update_features(self, state):
        ## clause features
        self.g.nodes['clause'].data['features'] = torch.cat([self.g.nodes['clause'].data['value'],
                                                              ],1)

        ## variable features
        self.g.nodes['variable'].data['features'] = torch.cat([state['steps_since_last_visited'].reshape(-1,1),
                                                               # state['immediate_cut_change'].reshape(-1,1),
                                                               # state['sat_on_change'].reshape(-1,1),
                                                               self.g.nodes['variable'].data['sat_value'].reshape(-1,1),
                                                              ],1)
        
        ## context features
        if self.args.drop_feats:
            feat = torch.tensor(0.)
        else:
            feat = (1-state['best_sat_value'])


        self.g.nodes['context'].data['features'] = torch.stack([#state['current_sat_value'],
                                                              state['difference_to_best_observed'],
                                                              state['remaining_steps'],
                                                              state['distance_to_best'],
                                                              feat,
                                                              # state['increasing_actions'],
                                                             ],dim=0).reshape(1,-1)
        
        

    
    
    def evaluate_neighbours(self, state):
        state['is_local_maxima'] = torch.tensor(True)        
        state['immediate_cut_change'] = torch.zeros(self.num_var)
        state['sat_on_change'] = torch.zeros(self.num_var)
        state['increasing_actions'] = torch.tensor(0.)
        
        for i in range(self.num_var):
            temp_state = self.copy_state(state)
            temp_state['step_update'] = torch.tensor(False)
            temp_state['assignment'][i] = 1-temp_state['assignment'][i]
            self.propagate_clause_values(temp_state)
            self.set_sat_values(temp_state)
            
            state['immediate_cut_change'][i] = temp_state['current_sat_value'] - state['current_sat_value']
            
            if temp_state['current_sat_value'] > state['current_sat_value']:
                state['is_local_maxima'] = torch.tensor(False)
                state['increasing_actions'] += 1

            if temp_state['current_sat_value'] == 1.:
                state['sat_on_change'][i] = 1
        
        
        self.propagate_clause_values(state)

        if state['is_local_maxima'] :
            for prev_local_max in self.local_maxima_set:
                if torch.equal(prev_local_max, state['assignment']):
                    state['is_local_maxima'] = torch.tensor(False)
                    break
        
        if state['is_local_maxima']:
            self.local_maxima_set.append(state['assignment'])
            
        
        return state
        
        
    def set_random_assignment(self):
        state = {}
        state['assignment'] = torch.bernoulli(torch.full([self.num_var], 0.5, dtype=torch.float32))
        state['best_sat_value'] = torch.tensor(0.)
        state['remaining_steps'] = torch.tensor(2*self.num_var).float()
        state['steps_since_last_visited'] = torch.zeros(self.num_var)
        # state['immediate_cut_change'] = torch.zeros(self.num_var)
        # state['sat_on_change'] = torch.zeros(self.num_var)
        state['is_local_maxima'] = torch.tensor(False)
        # state['clause_last_change'] = torch.zeros(self.num_clause)
        self.local_maxima_set = []
        self.propagate_clause_values(state)
        state = self.set_sat_values(state)

        # state = self.evaluate_neighbours(state)
        self.update_features(state)
        self.state = state

#     def set_state(self, state):
#         self.propagate_clause_values(state)
#         self.update_features(state)
        
        
    def copy_state(self, state):
        if state is None:
            return None
        
        new_state = {}
        
        for key in state.keys():
            new_state[key] = state[key].clone().detach()
        
        return new_state
        
        
    def get_graph(self):
        return self.g
    
    
    def get_graph_copy(self):

        sub_g = self.g.subgraph({'variable': torch.arange(self.num_var), 
                                 'clause':  torch.arange(self.num_clause), 
                                 'context': [0]})
#         sub_g.copy_from_parent()
        if self.args.device == 'cpu':
            sub_g.nodes['variable'].data.pop('embeddings')
            sub_g.nodes['variable'].data.pop('ctx')
            sub_g.nodes['clause'].data.pop('embeddings')
            sub_g.nodes['clause'].data.pop('ctx')
            sub_g.nodes['context'].data.pop('embeddings')
            sub_g.nodes['context'].data.pop('v')
            sub_g.nodes['context'].data.pop('c')

        sub_g.nodes['variable'].data.pop('value')
        sub_g.nodes['variable'].data.pop('stack')
        
        # sub_g.nodes['variable'].data.pop('h')
        sub_g.nodes['clause'].data.pop('value')
        # sub_g.nodes['clause'].data.pop('h')
    
        sub_g.edges['assigns'].data.pop('value')
        sub_g.edges['contains'].data.pop('value') 

        return sub_g
    
    
    def get_reward(self, best_sat_value):
        reward = max(torch.tensor(0.), self.state['current_sat_value'] - best_sat_value).detach()
        
        # if self.args.rew_on_lm and self.state['is_local_maxima'] and not self.state['current_sat_value'] == 1:
        #     if self.args.lm_reward_type == 'zero':
        #         reward = torch.tensor(0.)
        #     elif self.args.lm_reward_type == 'num_clause':
        #         reward += 1/self.num_clause
        
        if self.args.rew_on_sat and self.state['current_sat_value'] == 1.:
            if self.args.sat_reward_type == 'one':
                reward += 1
            elif self.args.sat_reward_type == 'num_clause':
                reward += 1/self.num_clause
            
        return reward.to(self.args.device)
    
    
    def step(self, action):
        best_sat_value = self.state['best_sat_value'].clone().detach()
        # prev_clause_values = self.g.nodes['clause'].data['value'][:,0].clone().detach()

        self.state['assignment'][action] = 1-self.state['assignment'][action]
        self.state['remaining_steps'] -= 1
        self.state['steps_since_last_visited'] += 1
        # self.state['clause_last_change'] += 1
        self.state['steps_since_last_visited'][action] = 0
        
        self.propagate_clause_values(self.state)

        # mask = ~torch.eq(prev_clause_values, self.g.nodes['clause'].data['value'][:,0])
        # self.state['clause_last_change'][mask] = 0
        # self.state['clause_last_change'] = torch.mul(self.state['clause_last_change'], self.g.nodes['clause'].data['value'][:,1])

        self.set_sat_values(self.state)
        # self.evaluate_neighbours(self.state)
        self.update_features(self.state)
        
        obs_reward = self.state['current_sat_value']
        reward = self.get_reward(best_sat_value)        
        done = obs_reward == 1.


        return obs_reward, reward, done
    



class SatEnvLookahead():
    def __init__(self, adjacency, args):
        self.args = args
        self.num_var = adjacency.shape[0]
        self.num_clause = adjacency.shape[1]
        self.backtrack = False
        
        edge_index = (adjacency != 0).nonzero().T
                
        graph_data = {('variable', 'assigns', 'clause'): (edge_index[0], edge_index[1]),
                       ('clause', 'contains', 'variable'): (edge_index[1], edge_index[0]),
                       ('variable', 'has_context', 'context'): (torch.arange(self.num_var, dtype=torch.int64), torch.zeros(self.num_var, dtype=torch.int64)),
                       ('clause', 'has_context', 'context'): (torch.arange(self.num_clause, dtype=torch.int64), torch.zeros(self.num_clause, dtype=torch.int64)),
                       ('context', 'has_node', 'clause'): (torch.zeros(self.num_clause, dtype=torch.int64), torch.arange(self.num_clause, dtype=torch.int64)),
                       ('context', 'has_node', 'variable'): (torch.zeros(self.num_var, dtype=torch.int64), torch.arange(self.num_var, dtype=torch.int64)),}
        
        self.g = dgl.heterograph(graph_data)
        self.state = None
        
        ## add edge weight as one-hot
        self.g.edges['contains'].data['value'] = F.relu(torch.tensor([1,-1]).reshape(2,1) * adjacency[tuple(edge_index)]).T
        self.g.edges['assigns'].data['value'] = F.relu(torch.tensor([1,-1]).reshape(2,1) * adjacency[tuple(edge_index)]).T

        self.var_degrees = torch.tensor([self.g.in_degree(i,etype='contains') for i in range(self.num_var)])



        
    
    def propagate_clause_values(self, state):
        var_assignment = state['assignment']
        self.g.nodes['variable'].data['value'] =  torch.stack([var_assignment, (1-var_assignment)]).T
        self.g.apply_nodes(lambda node: {'stack': torch.stack([node.data['value'],(1-node.data['value'])], dim=1)}, ntype='variable')
        
        ## assigns edge features
        self.g.apply_edges(fn.u_dot_e('stack', 'value', 'edge_sat'), etype='assigns')
        self.g.apply_edges(fn.v_dot_e('stack', 'value', 'edge_sat'), etype='contains')


        self.g.edges['assigns'].data['edge_sat'] = self.g.edges['assigns'].data['edge_sat'].reshape(-1,2)
        self.g.edges['contains'].data['edge_sat'] = self.g.edges['assigns'].data['edge_sat'].reshape(-1,2)
                
        # clause node features
        self.g.multi_update_all({
                    ('variable', 'assigns', 'clause'): (lambda edges: {'m': edges.data['edge_sat'][:,0]}, fn.max('m', 'value'))},
                   "max")
        

        self.g.nodes['clause'].data['value'] = torch.stack((self.g.nodes['clause'].data['value'],(1-self.g.nodes['clause'].data['value'])),1)

        self.g.multi_update_all({
                    ('clause', 'contains', 'variable'): (self.message_temp_clause_vals, fn.mean('m', 'sat_value'))}, 'mean')

        # print(self.g.nodes['variable'].data['sat_sum'])
        # print(torch.sum(self.g.nodes['variable'].data['sat_sum']))
        # error
        # self.g.nodes['variable'].data['sat_value'] = torch.div(self.g.nodes['variable'].data['sat_sum'], self.var_degrees)
        


        
        # self.g.multi_update_all({
        #             ('variable', 'has_context', 'context'): (self.message_temp_clause_vals, fn.mean('val', 'val2'))})
    

    def message_temp_clause_vals(self, edges):
        # print(edges.src['value'])
        # error
        return {'m' : edges.src['value'][:,0]}


#         ## TEMP!! 
#         self.g.multi_update_all({
#                     ('clause', 'contains', 'variable'): (self.message_temp_clause_vals, fn.mean('val', 'val2'))}
            
#         self.g.multi_update_all({
#                     ('variable', 'has_context', 'context'): (self.message_temp_clause_vals, fn.mean('val', 'val2')),
    
#     def message_temp_clause_vals(self, edges):
#         return {'val' : edges.src['value'][0]}
            
#     def message_temp_clause_vals(self, edges):
#         return {'val' : edges.src['value'][0]}
            
    
    
    def get_sat_value(self):
        if self.args.reward_type == 'clause':
            clause_values = self.g.nodes['clause'].data['value'][:,0]
            sat_value = torch.sum(clause_values)/clause_values.shape[0]
        elif self.args.reward_type == 'variable':
            sat_value = torch.sum(self.g.nodes['variable'].data['sat_value'])/self.num_var
        return sat_value



    
    def set_sat_values(self, state):
        state['current_sat_value'] = self.get_sat_value()
        state['is_increasing'] = state['current_sat_value'] > state['best_sat_value']
        
        if state['is_increasing']:
            state['best_sat_value'] = state['current_sat_value']
            state['best_assignment'] = state['assignment']
        
        state['difference_to_best_observed'] = state['best_sat_value'] - state['current_sat_value']
        state['distance_to_best'] = self.num_var - torch.sum(torch.eq(state['assignment'], state['best_assignment'])).float()
        
        return state
      
    
    def update_features(self, state):
        ## clause features
        self.g.nodes['clause'].data['features'] = torch.cat([self.g.nodes['clause'].data['value'],
                                                              ],1)

        ## variable features
        self.g.nodes['variable'].data['features'] = torch.cat([state['steps_since_last_visited'].reshape(-1,1),
                                                               state['immediate_cut_change'].reshape(-1,1),
                                                               state['sat_on_change'].reshape(-1,1),
                                                               self.g.nodes['variable'].data['sat_value'].reshape(-1,1),
                                                              ],1)
        
        ## context features
        self.g.nodes['context'].data['features'] = torch.stack([#state['current_sat_value'],
                                                              state['difference_to_best_observed'],
                                                              state['remaining_steps'],
                                                              state['distance_to_best'],
                                                              (1-state['best_sat_value']),
                                                              state['increasing_actions'],
                                                             ],dim=0).reshape(1,-1)
        
        

    
    
    def evaluate_neighbours(self, state):
        state['is_local_maxima'] = torch.tensor(True)        
        state['immediate_cut_change'] = torch.zeros(self.num_var)
        state['sat_on_change'] = torch.zeros(self.num_var)
        state['increasing_actions'] = torch.tensor(0.)
        
        for i in range(self.num_var):
            temp_state = self.copy_state(state)
            temp_state['step_update'] = torch.tensor(False)
            temp_state['assignment'][i] = 1-temp_state['assignment'][i]
            self.propagate_clause_values(temp_state)
            self.set_sat_values(temp_state)
            
            state['immediate_cut_change'][i] = temp_state['current_sat_value'] - state['current_sat_value']
            
            if temp_state['current_sat_value'] > state['current_sat_value']:
                state['is_local_maxima'] = torch.tensor(False)
                state['increasing_actions'] += 1

            if temp_state['current_sat_value'] == 1.:
                state['sat_on_change'][i] = 1
        
        
        self.propagate_clause_values(state)

        if state['is_local_maxima'] :
            for prev_local_max in self.local_maxima_set:
                if torch.equal(prev_local_max, state['assignment']):
                    state['is_local_maxima'] = torch.tensor(False)
                    break
        
        if state['is_local_maxima']:
            self.local_maxima_set.append(state['assignment'])
            
        
        return state
        
        
    def set_random_assignment(self):
        state = {}
        state['assignment'] = torch.bernoulli(torch.full([self.num_var], 0.5, dtype=torch.float32))
        state['best_sat_value'] = torch.tensor(0.)
        state['remaining_steps'] = torch.tensor(2*self.num_var).float()
        state['steps_since_last_visited'] = torch.zeros(self.num_var)
        state['immediate_cut_change'] = torch.zeros(self.num_var)
        state['sat_on_change'] = torch.zeros(self.num_var)
        state['is_local_maxima'] = torch.tensor(False)
        # state['clause_last_change'] = torch.zeros(self.num_clause)
        self.local_maxima_set = []
        self.propagate_clause_values(state)
        state = self.set_sat_values(state)

        state = self.evaluate_neighbours(state)
        self.update_features(state)
        self.state = state

#     def set_state(self, state):
#         self.propagate_clause_values(state)
#         self.update_features(state)
        
        
    def copy_state(self, state):
        if state is None:
            return None
        
        new_state = {}
        
        for key in state.keys():
            new_state[key] = state[key].clone().detach()
        
        return new_state
        
        
    def get_graph(self):
        return self.g
    
    

    
    
    def get_graph_copy(self):

        sub_g = self.g.subgraph({'variable': torch.arange(self.num_var), 
                                 'clause':  torch.arange(self.num_clause), 
                                 'context': [0]})
#         sub_g.copy_from_parent()
        if self.args.device == 'cpu':
            sub_g.nodes['variable'].data.pop('embeddings')
            sub_g.nodes['variable'].data.pop('ctx')
            sub_g.nodes['clause'].data.pop('embeddings')
            sub_g.nodes['clause'].data.pop('ctx')
            sub_g.nodes['context'].data.pop('embeddings')
            sub_g.nodes['context'].data.pop('v')
            sub_g.nodes['context'].data.pop('c')

        sub_g.nodes['variable'].data.pop('value')
        sub_g.nodes['variable'].data.pop('stack')
        
        # sub_g.nodes['variable'].data.pop('h')
        sub_g.nodes['clause'].data.pop('value')
        # sub_g.nodes['clause'].data.pop('h')
    
        sub_g.edges['assigns'].data.pop('value')
        sub_g.edges['contains'].data.pop('value') 

        return sub_g
    
    
    def get_reward(self, best_sat_value):
        reward = max(torch.tensor(0.), self.state['current_sat_value'] - best_sat_value).detach()
        
        if self.args.rew_on_lm and self.state['is_local_maxima'] and not self.state['current_sat_value'] == 1:
            if self.args.lm_reward_type == 'zero':
                reward = torch.tensor(0.)
            elif self.args.lm_reward_type == 'num_clause':
                reward += 1/self.num_clause
        
        if self.args.rew_on_sat and self.state['current_sat_value'] == 1.:
            if self.args.sat_reward_type == 'one':
                reward += 1
            elif self.args.sat_reward_type == 'num_clause':
                reward += 1/self.num_clause
            
        return reward.to(self.args.device)
    
    
    def step(self, action):
        best_sat_value = self.state['best_sat_value'].clone().detach()
        # prev_clause_values = self.g.nodes['clause'].data['value'][:,0].clone().detach()

        self.state['assignment'][action] = 1-self.state['assignment'][action]
        self.state['remaining_steps'] -= 1
        self.state['steps_since_last_visited'] += 1
        # self.state['clause_last_change'] += 1
        self.state['steps_since_last_visited'][action] = 0
        
        self.propagate_clause_values(self.state)

        # mask = ~torch.eq(prev_clause_values, self.g.nodes['clause'].data['value'][:,0])
        # self.state['clause_last_change'][mask] = 0
        # self.state['clause_last_change'] = torch.mul(self.state['clause_last_change'], self.g.nodes['clause'].data['value'][:,1])

        self.set_sat_values(self.state)
        self.evaluate_neighbours(self.state)
        self.update_features(self.state)
        
        obs_reward = self.state['current_sat_value']
        reward = self.get_reward(best_sat_value)        
        done = obs_reward == 1.


        return obs_reward, reward, done
    










