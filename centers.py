import torch


def get_center(dims):
    center = torch.randn((dims))
    center = center / torch.norm(center)
    return center


def get_distances(centers, point):
    return torch.norm(centers - point, dim=1)


def get_optimal_centers(dims, num_points, best_of=10000):
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
