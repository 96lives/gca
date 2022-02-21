from .chamfer_distance import ChamferDistance


def compute_chamfer_dist(c1, c2):
    chamfer_dist = ChamferDistance()
    dist1, dist2 = chamfer_dist(c1, c2)
    return (dist1.sqrt().mean().item() + dist2.sqrt().mean().item()) / 2


