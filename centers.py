import torch


def get_distances(centers, point, dim=1):
    return torch.norm(centers - point, dim=dim)


def get_classes(centers, center_mappings, features):
    classes = torch.zeros((features.shape[0]))
    for i in range(features.shape[0]):
        classes[i] = center_mappings[torch.argmin(
            get_distances(centers, features[i, ...]))]
    return classes
