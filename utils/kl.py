import torch


# code from https://github.com/pytorch/pytorch/issues/15288
def bernoulli_kl(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
	eps = 1e-10

	logq1 = (q + eps).log()
	logq0 = (1 - q + eps).log()
	logp1 = (p + eps).log()
	logp0 = (1 - p + eps).log()

	kl1 = q * (logq1 - logp1)
	kl0 = (1 - q) * (logq0 - logp0)
	return kl1 + kl0


def gaussian_kl(
		mu_q: torch.Tensor,
		mu_p: torch.Tensor,
		sigma: torch.Tensor
) -> torch.Tensor:
	"""
	:param mu_q: tensor of size N x K
	:param mu_p: tensor of size N x K
	:param sigma: tensor of size N or scalar
	:return: N tensor
	"""
	delta = (mu_q - mu_p) ** 2
	return delta.sum(dim=1) / (2 * (sigma ** 2))
