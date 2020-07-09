import torch
from ignite.metrics import Metric


class ClusteringLoss():
    def __init__(self, encoding_weight, decoding_weight, clustering_weight):
        self.encoding_weight = encoding_weight
        self.decoding_weight = decoding_weight
        self.clustering_weight = clustering_weight
        self.mse = torch.nn.MSELoss()

    def __call__(self, models, device, state):
        encoded = models[0](state['data'])
        decoded = models[1](encoded)
        loss = self.mse(decoded[..., 2:-2, 2:-2], state['data'])
        state['dec_loss'] = loss


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
    def __init__(self, model_name, prefix):
        self.loss = []
        super(Loss, self).__init__(model_name, prefix + 'loss')

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
