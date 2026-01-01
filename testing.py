import torch, math
from acpl.baselines.coins import su2_from_zyz, fixed_coin_su2_theta, hadamard_su2

U1 = su2_from_zyz(fixed_coin_su2_theta("hadamard"))
U2 = hadamard_su2()
print("hadamard max abs diff:", (U1 - U2).abs().max().item())

Ug = su2_from_zyz(fixed_coin_su2_theta("grover"))
X = torch.tensor([[0.,1.],[1.,0.]])
print("grover max abs diff:", (Ug - (1j*X.to(torch.complex64))).abs().max().item())

# unitary check
I = torch.eye(2, dtype=U1.dtype)
print("unitary error:", (U1.conj().T @ U1 - I).abs().max().item(), "det:", torch.linalg.det(U1))
