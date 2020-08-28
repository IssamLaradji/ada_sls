import torch
import time
import tqdm
from .cg import CG


def cg_hvp_solve(params, grad_current, closure, alpha, max_iter=None, 
            rtol=1e-8, atol=1e-6, verbose=0, 
            negcurv_flag=False, sanity_flag=False):
    loss = closure()
    A_bmm_mark = lambda p: hvp_model(params, loss, p, alpha=alpha, 
                                     retain_graph=True)
    if sanity_flag:
        n_params = sum([p.view(-1).size()[0] for p in params])
        V = torch.randn(n_params).cuda()
        assert A_bmm_mark(V).sum() ==  A_bmm(V[None]).sum()
    
    # Stack the gradients for cg
    b = torch.cat([gc.view(-1) for gc in grad_current])
    dk = CG(A_bmm=A_bmm_mark, b=b, optTol=rtol, verbose=1, maxIter=100)
    
    return dk

@torch.enable_grad()
def hvp_model(params, loss, V, retain_graph=False, alpha=0, return_G=False):
    V = V.view(-1)
    n_params = V.shape[0]
    gv = 0.
    G = torch.zeros(n_params).cuda()
    s_ind = 0
    for p in params:
        g = torch.autograd.grad(loss, p, retain_graph=True, create_graph=True)[0].view(-1)
        n_params_layer = g.shape[0]

        s, e = s_ind, s_ind + n_params_layer
        G[s:e] = g
        gv += (g * V[s:e]).sum()

        s_ind += n_params_layer

    Hv = torch.zeros(n_params).cuda()
    s_ind = 0
    for p in params:
        # v.requires_grad = False
        Hv_layer = torch.autograd.grad(gv, p, retain_graph=True,
                            create_graph=False)[0]
        Hv_layer = Hv_layer.view(-1)
        n_params_layer = Hv_layer.shape[0]
        s, e = s_ind, s_ind + n_params_layer

        Hv_layer += alpha * V[s:e]
        Hv[s:e] = Hv_layer

        s_ind += n_params_layer

    if return_G:
        return Hv, G

    return Hv