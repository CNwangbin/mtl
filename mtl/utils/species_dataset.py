import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class SpeciesEmbeddingDataset(Dataset):
    
    def __init__(self, data_file, species_dict):
        super().__init__()
        self.df = pd.read_pickle(data_file)
        self.species_dict = species_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        idx_df = self.df.iloc[[idx]]
        embeddings, labels = self.__data2ternsor(idx_df, self.species_dict)
        encoded_inputs = {'embeddings': embeddings, 'labels': labels}
        return encoded_inputs
    
    def __data2ternsor(self, df, species_dict):
        embeddings = torch.from_numpy(np.array(df['embeddings'].values.item(), dtype=np.float32))
        labels = np.zeros((len(df), len(species_dict)), dtype=np.int32)
        for i, row in enumerate(df.itertuples()):
            for spe in row.species:
                if spe in species_dict:
                    labels[i, species_dict[spe]] = 1
        labels = torch.from_numpy(labels).int()
        return embeddings, labels
    


if __name__ == '__main__':
    test_data = '../../data/test_data_embedding.pkl'
    species_data = '../../data/species.pkl'
    species_map = {v:i for i,v in enumerate(pd.read_pickle(species_data).species.values.flatten())}
    nb_species = len(species_map)
    # Dataset and DataLoader
    val_dataset = SpeciesEmbeddingDataset(test_data, species_map)
    # dataloder
    test_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )