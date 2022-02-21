import torch

from torch.utils.cpp_extension import load
cd = load(
    name="cd",
    sources=[
        "utils/chamfer_distance/chamfer_distance.cpp",
        "utils/chamfer_distance/chamfer_distance.cu"
    ],
    build_directory='./'
)


class ChamferDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2, return_idx):

        # 2 dimensional
        if xyz1.shape[2] == 2:
            xyz1 = torch.cat([xyz1, torch.zeros(1, xyz1.shape[1], 1).to(xyz1.device)], dim=2)
            xyz2 = torch.cat([xyz2, torch.zeros(1, xyz2.shape[1], 1).to(xyz2.device)], dim=2)

        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n, dtype=torch.int)
        idx2 = torch.zeros(batchsize, m, dtype=torch.int)

        if not xyz1.is_cuda:
            cd.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        else:
            dist1 = dist1.cuda()
            dist2 = dist2.cuda()
            idx1 = idx1.cuda()
            idx2 = idx2.cuda()
            cd.forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)

        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        if return_idx:
            return dist1, dist2, idx1, idx2
        return dist1, dist2

    @staticmethod
    def backward(ctx, graddist1, graddist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors

        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        if not graddist1.is_cuda:
            cd.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        else:
            gradxyz1 = gradxyz1.cuda()
            gradxyz2 = gradxyz2.cuda()
            cd.backward_cuda(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)

        return gradxyz1, gradxyz2


class ChamferDistance(torch.nn.Module):
    def forward(self, xyz1, xyz2, return_idx=False):
        return ChamferDistanceFunction.apply(xyz1, xyz2, return_idx)

