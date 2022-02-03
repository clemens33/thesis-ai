
import matplotlib.pyplot as plt
import torch
from entmax import sparsemax as entmax_sparsemax
from entmax import entmax15 as entmax_entmax15
from entmax import entmax_bisect

def softmax(z):
    z = torch.stack([z, z], dim=1)
    z[:, 0] = 0.0

    o = torch.softmax(z, dim=-1)

    return o[:, 1]

def sparsemax(z):
    z = torch.stack([z, z], dim=1)
    z[:, 0] = 0.0

    o = entmax_sparsemax(z)

    return o[:, 1]

def alpha_entmax(z, alpha):
    z = torch.stack([z, z], dim=1)
    z[:, 0] = 0.0

    o = entmax_bisect(z, alpha)

    return o[:, 1]

def entmax15(z):
    z = torch.stack([z, z], dim=1)
    z[:, 0] = 0.0

    o = entmax_entmax15(z)

    return o[:, 1]

z = torch.linspace(-5, 5, 10000, requires_grad=True)

sm = softmax(z)
sm.sum().backward()
sm_grad = z.grad.clone()
z.grad = None

sp = sparsemax(z)
sp.sum().backward()
sp_grad = z.grad.clone()
z.grad = None

em15 = entmax15(z)
em15.sum().backward()
em15_grad = z.grad.clone()
z.grad = None

em125 = alpha_entmax(z, 1.25)
em125.sum().backward()
em125_grad = z.grad.clone()
z.grad = None

em4 = alpha_entmax(z, 4.0)
em4.sum().backward()
em4_grad = z.grad.clone()
z.grad = None

em3 = alpha_entmax(z, 3.0)
em3.sum().backward()
em3_grad = z.grad.clone()
z.grad = None

#z = torch.linspace(0, 1, 100, requires_grad=True)
z = torch.rand(size=(10, ), requires_grad=True)
sm = torch.softmax(z, dim=-1)
sm.sum().backward()
print(z.grad.max())
print(z.grad.min())


z = torch.linspace(0, 1, 100, requires_grad=True)
em_3 = entmax_bisect(z, 1.5)
em_3.backward(torch.ones_like(z))
print(z.grad.max())
print(z.grad.min())

#

# z = z.detach().numpy()
#
# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
#
# ax1.plot(z, sm.detach(), "--", alpha=0.5, label=r"$\alpha=1$")
# ax1.plot(z, em125.detach(), alpha=0.5, label=r"$\alpha=1.25$")
# ax1.plot(z, em15.detach(), alpha=0.5, label=r"$\alpha=1.5$")
# ax1.plot(z, sp.detach(), ":", alpha=0.5, label=r"$\alpha=2$")
# ax1.plot(z, em3.detach(), alpha=0.5, label=r"$\alpha=3$")
# ax1.plot(z, em4.detach(), alpha=0.5, label=r"$\alpha=4$")
# ax1.set_ylabel(r"$entmax_{\alpha}([t,0])_1$")
# ax1.set_xlabel("t")
# #ax1.set_title(r"$entmax_{\alpha}([t,0])_1$ evaluated")
# ax1.legend()
# ax1.grid(linewidth=0.3)
#
#
# ax2.plot(z, sm_grad, "--", alpha=0.5, label=r"$\alpha=1$")
# ax2.plot(z, em125_grad, alpha=0.5, label=r"$\alpha=1.25$")
# ax2.plot(z, em15_grad, alpha=0.5, label=r"$\alpha=1.5$")
# ax2.plot(z, sp_grad, ":", alpha=0.5, label=r"$\alpha=2$")
# ax2.plot(z, em3_grad, alpha=0.5, label=r"$\alpha=3$")
# ax2.plot(z, em4_grad, alpha=0.5, label=r"$\alpha=4$")
# ax2.set_xlabel("t")
# ax2.set_ylabel(r"$\frac{1}{\partial t} \partial entmax_{\alpha}([t,0])_1$")
# #ax2.set_title(r"gradient of $entmax_{\alpha}([t,0])_1$ w.r.t $t$")
# ax2.legend()
# ax2.grid(linewidth=0.3)
#
# plt.tight_layout(pad=1.0)
#
# plt.savefig("entmax.png")
# plt.show()