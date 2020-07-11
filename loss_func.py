import torch
import numpy as np
from ignite.metrics import Metric
from data import augumentation_transform
from centers import get_distances, get_mappping, get_classes
import matplotlib.pyplot as plt


class ClusteringLoss():
    def __init__(self, centers, labeled_data, dims, classes, encoding_weight,
                 decoding_weight, clustering_weight):
        self.centers = centers[None, ...]
        self.labeled_data = labeled_data
        self.dims = dims
        self.classes = classes
        self.encoding_weight = encoding_weight
        self.decoding_weight = decoding_weight
        self.clustering_weight = clustering_weight
        self.mse = torch.nn.MSELoss()

    def __call__(self, models, device, state):
        data, data_aug = state['data']
        encoded = models[0](data)
        encoded_aug = models[0](data_aug)
        decoded = models[1](encoded_aug)
        reconstruction_loss = self.mse(decoded[..., 2:-2, 2:-2], data)
        feature_loss = self.mse(encoded_aug, encoded)
        # cluster_loss = mmd_rbf(encoded_aug, self.centers[0, ...])
        cluster_loss = (
            torch.sum(get_distances(self.centers, encoded[:, None, :], dim=2))
            + torch.sum(
                get_distances(self.centers, encoded_aug[:, None, :], dim=2))
        ) / (2 * data.shape[0] * self.classes * self.dims)

        mapping = get_mappping(self.centers[0, ...].cpu(),
                               models[0](self.labeled_data[0]).detach().cpu(),
                               self.labeled_data[1].cpu())
        state['mapping'] = mapping
        state['features'] = encoded.detach().cpu()
        state[models[0].get_name() +
              '_feature_loss'] = feature_loss.detach().cpu().item()
        state[models[-1].get_name() +
              '_recon_loss'] = reconstruction_loss.detach().cpu().item()
        state[models[0].get_name() +
              '_cluster_loss'] = cluster_loss.detach().cpu().item()
        # # self.clustering_weight * cluster_loss +

        # #
        # loss = (self.decoding_weight * reconstruction_loss)
        return self.decoding_weight * reconstruction_loss + self.clustering_weight * cluster_loss + feature_loss * self.encoding_weight


class BaseMetric(Metric):
    def __init__(self, model_name, metric_name):
        self.model_name = model_name
        self.metric_name = metric_name
        super(BaseMetric, self).__init__(output_transform=lambda x: x)

    def get_metric_name(self):
        return self.metric_name

    def get_model_name(self):
        return self.model_name


class Loss(BaseMetric):
    def __init__(self, model_name, prefix=''):
        super(Loss, self).__init__(model_name, prefix + 'loss')
        self.reset()

    def reset(self):
        self.loss = []

    def update(self, output):
        loss = output[self.get_model_name() + '_loss']
        if loss is not None:
            self.loss.append(loss)

    def compute(self):
        result = np.nan if len(
            self.loss) == 0 else sum(self.loss) / len(self.loss)
        return result


class Accuracy(BaseMetric):
    def __init__(self, model_name, centers):
        self.centers = centers
        super(Accuracy, self).__init__(model_name, 'acc')
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, output):
        features = output['features']
        plt.scatter(self.centers[:, 0], self.centers[:, 1], c='red')
        plt.scatter(features[:, 0], features[:, 1], c='blue')
        plt.savefig('tmp.png')
        classes = get_classes(self.centers, output['mapping'],
                              output['features'])
        self.correct += torch.sum(classes == output['labels'].cpu())
        self.total += output['labels'].shape[0]

    def compute(self):
        return float(self.correct) / self.total
