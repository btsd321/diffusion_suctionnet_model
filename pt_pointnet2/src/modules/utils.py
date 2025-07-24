import torch

def farthest_point_sample(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    远thest点采样函数，从给定的点云中选择最远的点。
    
    参数:
        points: 输入点云，形状为 (B, N, C)，B为批量大小，N为点的数量，C为每个点的特征维度。
        num_samples: 需要采样的点的数量。
    
    返回:
        torch.Tensor: 采样后的点的索引，形状为 (B, num_samples)。
    """
    B, N, _ = points.shape
    centroids = torch.zeros(B, num_samples).long().to(points.device)
    distance = torch.ones(B, N).to(points.device) * 1e10
    farthest = torch.randint(0, N, (B,)).to(points.device)
    
    for i in range(num_samples):
        centroids[:, i] = farthest
        centroid = points[torch.arange(B), farthest].view(B, 1, -1)
        dist = torch.sum((points - centroid) ** 2, dim=-1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]
    
    return centroids

def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    根据给定的索引从点云中选择点。
    
    参数:
        points: 输入点云，形状为 (B, N, C)。
        idx: 选择的点的索引，形状为 (B, S)。
    
    返回:
        torch.Tensor: 选择后的点云，形状为 (B, S, C)。
    """
    B = points.shape[0]
    return points[torch.arange(B).view(B, 1, 1), idx.view(B, -1, 1).expand(-1, -1, points.shape[2])]

def get_graph_feature(points: torch.Tensor, k: int, idx: torch.Tensor) -> torch.Tensor:
    """
    获取图形特征，基于k近邻关系。
    
    参数:
        points: 输入点云，形状为 (B, N, C)。
        k: 每个点的邻居数量。
        idx: 邻居的索引，形状为 (B, N, k)。
    
    返回:
        torch.Tensor: 图形特征，形状为 (B, N, C * (k + 1))。
    """
    B, N, C = points.shape
    idx = idx.view(B, N, k, 1).expand(-1, -1, -1, C)
    grouped_points = points[torch.arange(B).view(B, 1, 1), idx]
    points = points.view(B, N, 1, C).expand(-1, -1, k, -1)
    return torch.cat([points, grouped_points], dim=-1)  # (B, N, k, C * 2)