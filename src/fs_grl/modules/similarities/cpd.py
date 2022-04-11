import torch
from torchcpd.deformable_registration import DeformableRegistration


def cpd_distance(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute symmetric CPD distance between source and target

    Args:
        source: single point cloud with shape [k1, e]
        target: single point cloud with shape [k2, e]

    Returns:
        backpropagable distance between source and target
    """
    cpd = DeformableRegistration(X=target, Y=source)
    cpd.register()

    cpd_inv = DeformableRegistration(X=source, Y=target)
    cpd_inv.register()

    distance = torch.linalg.norm(cpd.G @ cpd.W) + torch.linalg.norm(cpd_inv.G @ cpd_inv.W)

    return distance


def cpd_similarity(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute symmetric CPD similarity between source and target

    Args:
        source: single point cloud with shape [k1, e]
        target: single point cloud with shape [k2, e]

    Returns:
        backpropagable similarity between source and target
    """
    distance: torch.Tensor = cpd_distance(source=source, target=target)
    return 1 / (1 + distance)
