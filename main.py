import sys
import logging
import torch

from data import get_data_generators
from trainer import ModelTrainer
from models import Encoder, Decoder
from utils import ClusteringLoss, Loss, Accuracy

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s: %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

DIMS = 5
CLASSES = 10
LR = 0.0005
BATCH_SIZE = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.getLogger().info('Using device: ' + str(device))

labeled_data, train, test = get_data_generators(BATCH_SIZE)
labeled_data = (labeled_data[0].to(device), labeled_data[1].to(device))

models = [Encoder(DIMS, 'enc').to(device), Decoder(DIMS, 'dec').to(device)]

optimizers = [
    torch.optim.Adam(models[0].parameters(), LR),
    torch.optim.Adam(models[1].parameters(), LR)
]

generators = {'train': train, 'test': test}

metrics = {
    'enc_train': [
        Loss('enc_feature', 'feature'),
        Loss('enc_cluster', 'cluster'),
    ],
    'dec_train': [
        Loss('total', 'total'),
        Loss('dec_recon', 'recon'),
    ],
    'enc_test': [
        Loss('enc_feature', 'feature'),
        Loss('enc_cluster', 'cluster'),
        Accuracy('enc')
    ],
    'dec_test': [
        Loss('total', 'total'),
        Loss('dec_recon', 'recon'),
    ]
}

trainer = ModelTrainer(models,
                       optimizers,
                       generators,
                       device,
                       ClusteringLoss(labeled_data, DIMS, CLASSES, 0.01, 1,
                                      0.01),
                       metrics,
                       log_step=1)

trainer.train(300)
