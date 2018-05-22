import torch
from timeit import default_timer as timer

A = torch.randn(416*128*3, 1).cuda()

E = torch.ones(1, 416*128*3).cuda()
t = timer()
c2 = E.mm(A.abs())
print(timer()-t)
print(c2)

t = timer()
c1 = A.abs().mean()
print(timer()-t)

t = timer()
b = A.abs()
c1 = b.mean()
print(timer()-t)

t = timer()
c1 = A.abs().mean()
print(timer()-t)

t = timer()
b = A.abs()
c1 = b.mean()
print(timer()-t)

t = timer()
c1 = A.abs().mean()
print(timer()-t)

t = timer()
b = A.abs()
c1 = b.mean()
print(timer()-t)

t = timer()
c1 = A.abs().mean()
print(timer()-t)

t = timer()
b = A.abs()
c1 = b.mean()
print(timer()-t)

t = timer()
c1 = A.abs().mean()
print(timer()-t)

t = timer()
b = A.abs()
c1 = b.mean()
print(timer()-t)
