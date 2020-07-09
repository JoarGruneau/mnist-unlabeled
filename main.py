import torch
from centers import get_optimal_centers
from data import get_data_generators
from trainer import ModelTrainer
from models import Encoder, Decoder
from loss_func import ClusteringLoss, Loss

DIMS = 60
CLASSES = 10
LR = 1e-3
BATCH_SIZE = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

centers = get_optimal_centers(DIMS, CLASSES)
train, test = get_data_generators(BATCH_SIZE)
models = [Encoder(DIMS, 'enc').to(device), Decoder(DIMS, 'dec').to(device)]

optimizers = [
    torch.optim.Adam(models[0].parameters(), LR),
    torch.optim.Adam(models[1].parameters(), LR)
]

generators = {'train': train, 'test': test}

metrics = {
    'enc_train': [],
    'dec_train': [Loss('dec', '')],
    'enc_test': [],
    'dec_test': [Loss('dec', '')]
}

trainer = ModelTrainer(models, optimizers, generators, device,
                       ClusteringLoss(1, 1, 1), metrics)

trainer.train(20)

# for x, y in train:
#     breakpoint()
