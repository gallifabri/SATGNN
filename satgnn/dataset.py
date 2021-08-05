import torch
import numpy as np

class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, directory):
        'Initialization'
    #         self.labels = labels
        self.directory = directory
        self.list_IDs = list_IDs
        self.size = len(self.list_IDs)

    def __len__(self):
        'Denotes the total number of samples'
        return self.size

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load(self.directory + str(ID) + '.pt')
    #         y = self.labels[ID]

        return X#, y


    def permute_samples(self):
        permutation = np.random.permutation(self.size)
        new_index = {permutation[i] : self.list_IDs[i] for i in range(self.size)}
        self.list_IDs = new_index




        
