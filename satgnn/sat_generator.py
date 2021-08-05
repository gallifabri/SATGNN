import random
from itertools import combinations
import numpy as np
import os
import torch
from dataset import Dataset
import math
import PyMiniSolvers.minisolvers as minisolvers
from pathlib import Path




### TODO 
# Comment unsat?
# figure out path to save
# save data
# arguments for dataset type
# add labels to dataset

from os import listdir
from os.path import isfile, join

class SatGenerator():
    #### SR SAT ####
    



    def generate_SR_sat(self):
        # Generates SAT examples only
        for pair in range(self.n_pairs):
            if pair % 100 == 0: print("[%d]" % pair)
            n_vars, iclauses, iclause_unsat, iclause_sat = self.gen_iclause_pair()
            out_filenames = self.mk_out_filenames(n_vars, pair)

            iclauses.append(iclause_unsat)
            # self.write_dimacs_to(n_vars, iclauses, out_filenames[0])

            iclauses[-1] = iclause_sat

            m = len(iclauses)
            adjacency = self.get_adjacency_matrix(iclauses, n_vars, m)
            tensor = torch.tensor(adjacency, dtype=torch.float32)
            torch.save(tensor, self.out_dir+str(pair)+'.pt')

            
            # self.write_dimacs_to(n_vars, iclauses, out_filenames[1])

    def write_dimacs_to(self, n_vars, iclauses, out_filename):
        with open(out_filename, 'w') as f:
            f.write("p cnf %d %d\n" % (n_vars, len(iclauses)))
            for c in iclauses:
                for x in c:
                    f.write("%d " % x)
                f.write("0\n")

    def mk_out_filenames(self, n_vars, t):
        prefix = "%s/sr_n=%.4d_pk2=%.2f_pg=%.2f_t=%d" % \
            (self.out_dir, n_vars, self.p_k_2, self.p_geo, t)
        return ("%s_sat=0.dimacs" % prefix, "%s_sat=1.dimacs" % prefix)

    def generate_k_iclause(self, n, k):
        vs = np.random.choice(n, size=min(n, k), replace=False)
        return [v + 1 if random.random() < 0.5 else -(v + 1) for v in vs]

    def gen_iclause_pair(self):
        n = random.randint(self.min_n, self.max_n)

        solver = minisolvers.MinisatSolver()
        for i in range(n): solver.new_var(dvar=True)

        iclauses = []
        ks = []

        while True:
            k_base = 1 if random.random() < self.p_k_2 else 2
            k = k_base + np.random.geometric(self.p_geo)
            # ks.append(k)
            iclause = self.generate_k_iclause(n, k)

            solver.add_clause(iclause)
            is_sat = solver.solve()
            if is_sat:
                iclauses.append(iclause)
            else:
                break

        # print('mean: ', np.mean(ks))
        iclause_unsat = iclause
        iclause_sat = [- iclause_unsat[0] ] + iclause_unsat[1:]
        return n, iclauses, iclause_unsat, iclause_sat


    def parse_dimacs(filename):
        print(filename)
        with open(filename, 'r') as f:
            lines = f.readlines()
        i = 0
        while lines[i].strip().split(" ")[0] == "c":
            i += 1
        header = lines[i].strip().split(" ")
        assert(header[0] == "p")
        n_vars = int(header[2])
        iclauses = [[int(s) for s in line.strip().split(" ")[:-1]] for line in lines[i+1:]]
        return n_vars, iclauses





    #### RANDOM K-SAT ####
    def generate_uniform_sat(self, m, k, n, forced_sat=True):
        #lower Case +ve
        positive_var = [i for i in range(1,n+1)]#(list(ascii_lowercase))[:n]
        negative_var = [-i for i in range(1,n+1)]#[c.upper() for c in positive_var]
        variables = positive_var + negative_var

        threshold = 10
        problem = []
        allCombs = list(combinations(variables, k))
        i = 0
        
        if forced_sat:
            solution = np.random.binomial(size=n, n=1, p= 0.5)
            solution = np.where(solution == 1, solution, -1)

        while i<m:
            c = random.sample(allCombs, 1)[0]
            tautology = False
            
            if forced_sat:
                j = k
                for l in c:
                    if solution[abs(l)-1] != (l/abs(l)):
                        j -= 1
                
                if j == 0:
                    continue
                
            
            if c not in problem:
                for j, l in enumerate(c):
                    for j2 in range(j+1,len(c)):
                        if c[j] + c[j2] == 0:
                            tautology = True
                            break
                    if tautology:
                        break
                        
                if not tautology:
                    i += 1
                    problem.append(list(c))
                
        print(problem)
        return problem



    def get_adjacency_matrix(self, problem, n, m):
        adj_m = np.zeros([n,m])
        
        for j in range(m):
            c = problem[j]
            
            for l in c:
                adj_m[abs(l)-1][j] = 1 if l > 0 else -1
        
        return adj_m


    def get_edge_indices(self, adj_m):
        edge_index = []
        
        for i, var_row in enumerate(adj_m):
            for j in range(len(var_row)):
                if adj_m[i][j] != 0:
                    edge_index.append((i,j))
                    
        return edge_index


    def make_data(self, directory, samples, m, k, n, forced_sat=True):
        # directory = os.pardir+'/data/'+('sat' if forced_sat else 'unsat') +'_s'+str(samples)+'_m'+str(m)+'_k'+str(k)+'_n'+str(n)+'/'
        # print(directory)
        
        
        

        for i in range(samples):
            sample = self.generate_uniform_sat(m, k, n, forced_sat=forced_sat)
            adjacency = self.get_adjacency_matrix(sample, n, m)
            tensor = torch.tensor(adjacency, dtype=torch.float32)
            torch.save(tensor, directory+str(i)+'.pt')


    def get_dataset(self, args, eval=False):
        forced_sat = args.forced_sat
        m = args.n * args.coef
        distribution = args.dataset
        train_idx = math.floor(args.samples * args.train_coef)
        val_idx = train_idx + math.floor(args.samples * args.val_coef)
        eval_idx = train_idx + val_idx

        data_prefix = 'eval_' if eval else ''


        directory = os.pardir+'/data/'+distribution+'_'+data_prefix+('sat' if forced_sat else 'unsat') +'_s'+str(args.samples)+'_m'+str(m)+'_k'+str(args.k)+'_n'+str(args.n)+'/'

        if not os.path.exists(directory):
            os.makedirs(directory)
            print('Generating data.')
            
            if distribution == 'ksat':
                self.make_data(directory, args.samples, m, args.k, args.n)
            else:
                self.p_k_2 = 0.3
                self.p_geo = 0.4
                self.min_n = args.n
                self.max_n = args.n
                self.n_pairs = args.samples
                self.out_dir = directory
                self.generate_SR_sat()
        
        list_IDs = [i for i in range(args.samples)]

        train_ids = list_IDs[0:train_idx]
        val_ids = list_IDs[train_idx:val_idx]
        eval_ids = list_IDs[val_idx:]


        return Dataset(train_ids,directory), Dataset(val_ids,directory), Dataset(eval_ids, directory)



    def get_eval_dataset(self, args):
        forced_sat = args.forced_sat
        m = args.n * args.coef
        distribution = args.dataset



        directory = os.pardir+'/data/'+distribution+'_eval_'+('sat' if forced_sat else 'unsat') +'_s'+str(args.samples)+'_m'+str(m)+'_k'+str(args.k)+'_n'+str(args.n)+'/'

        if not os.path.exists(directory):
            os.makedirs(directory)
            print('Generating data.')
            
            if distribution == 'ksat':
                self.make_data(directory, args.samples, m, args.k, args.n)
            else:
                self.p_k_2 = 0.3
                self.p_geo = 0.4
                self.min_n = args.n
                self.max_n = args.n
                self.n_pairs = args.samples
                self.out_dir = directory
                self.generate_SR_sat()
        
        list_IDs = [i for i in range(args.samples)]



        return Dataset(list_IDs,directory)
   

    def parse_dimacs(self, path):
        files = [f for f in listdir(path) if isfile(join(path, f))]
        directory = path+'_data/'

        if not os.path.exists(directory):
            os.makedirs(directory)

        for j, filename in enumerate(files):
            print(filename)
            with open(path+'/'+filename, 'r') as f:
                lines = f.readlines()

            i = 0
            while lines[i].strip().split(" ")[0] == "c":
                i += 1
            header = lines[i].strip().split(" ")
            assert(header[0] == "p")
            n_vars = int(header[2])
            iclauses = [[int(s) for s in line.strip().split(" ")[:-1]] for line in lines[i+1:]]

            m = len(iclauses)
            adjacency = self.get_adjacency_matrix(iclauses, n_vars, m)
            tensor = torch.tensor(adjacency, dtype=torch.float32)
            
            torch.save(tensor, directory+str(j)+'.pt')


#    ______            ______            ______
#   /\_____\          /\_____\          /\_____\          ____
#  _\ \__/_/_         \ \__/_/_         \ \__/_/         /\___\
# /\_\ \_____\        /\ \_____\        /\ \___\        /\ \___\
# \ \ \/ / / /        \ \/ / / /        \ \/ / /        \ \/ / /
#  \ \/ /\/ /          \/_/\/ /          \/_/_/          \/_/_/
#   \/_/\/_/              \/_/


         
