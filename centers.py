import torch


def get_center(dims):
    center = torch.randn((dims))
    center = center / torch.norm(center)
    return center


def get_distances(centers, point, dim=1):
    return torch.norm(centers - point, dim=dim)


def get_mappping(centers, mapping, labels):
    center_mapping = torch.zeros((centers.shape[0]))
    idx_to_map = [i for i in range(centers.shape[0])]
    free_mappings = [i for i in range(centers.shape[0])]
    while idx_to_map:
        min_distances = torch.zeros((len(idx_to_map)))
        mapping_idx = [None] * len(idx_to_map)
        for i, center_idx in enumerate(idx_to_map):
            distances = get_distances(mapping[free_mappings, :],
                                      centers[center_idx, ...])
            min_idx = torch.argmin(distances)
            mapping_idx[i] = min_idx
            min_distances[i] = distances[min_idx]
        best_fit = torch.argmin(min_distances)
        center_mapping[idx_to_map[best_fit]] = labels[free_mappings][
            mapping_idx[best_fit]]
        idx_to_map.pop(best_fit)
        free_mappings.pop(mapping_idx[best_fit])
    return center_mapping


def get_classes(centers, center_mappings, features):
    classes = torch.zeros((features.shape[0]))
    for i in range(features.shape[0]):
        classes[i] = center_mappings[torch.argmin(
            get_distances(centers, features[i, ...]))]
    return classes


def get_optimal_centers(dims, num_points, best_of=100000):
    points = [get_center(dims)]
    for _ in range(num_points - 1):
        candidates = []
        candidate_distances = torch.zeros((best_of))
        point_matrix = torch.stack(points, dim=0)
        for i in range(best_of):
            candidate_point = get_center(dims)
            candidates.append(candidate_point)
            candidate_distances[i] = torch.sum(
                get_distances(point_matrix, candidate_point))
        points.append(candidates[torch.argmax(candidate_distances)])
    return points
