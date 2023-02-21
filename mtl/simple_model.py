from torch import nn
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F

class SpeciesFC(nn.Module):
    def __init__(self, protein_dim, latent_dim, prediction_dim, dropout=0.):
        super().__init__()
        self.dropout = dropout
        self.nb_classes = prediction_dim
        self.fc = nn.Linear(protein_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, prediction_dim)

    def forward(self, embeddings, labels=None):
        embeddings = self.fc(embeddings)
        embeddings = F.relu(embeddings)
        embeddings = F.dropout(embeddings, self.dropout)
        logits = self.fc2(embeddings)
        outputs = (logits, )
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.nb_classes),
                            labels.view(-1, self.nb_classes))
            outputs = (loss, logits)

        return outputs
