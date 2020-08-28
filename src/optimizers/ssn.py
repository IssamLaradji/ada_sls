import sys
import copy
import traceback
import warnings

import torch
from torch.optim import Optimizer
from torch import autograd
import numpy as np

from .base import hvp
from .base import cg
from .base import utils as ut


class Ssn(Optimizer):
    def __init__(self, params, n_batches_per_epoch, init_step_size=1.0, 
    lr=None, c=0.1, beta=0.9, gamma=1.5, reset_option=1, lm=1e-3):
        params = list(params)
        super().__init__(params, {})

        self.params = params
        self.lr = lr # if lr is given, we don't perform line search

        # options for the line-search      
        self.n_batches_per_epoch = n_batches_per_epoch
        self.init_step_size = init_step_size       
        self.c = c
        self.beta = beta
        self.gamma = gamma
        self.reset_option = reset_option

        # additional regularization for solving the linear system
        self.lm = lm

        # things needed to be stored in the optimizer state
        self.state['step_size'] = init_step_size

    def step(self, closure):
        params_current = copy.deepcopy(self.params)
        # Instead of calling loss.backward(), we manually compute 
        # the gradients and make sure the corresponding graph is created, 
        # in order to compute the second derivatives
        d = sum([sum(p.shape) for p in params_current])

        loss = closure()
        grad_current = autograd.grad(loss, self.params, create_graph=True)

        if self.lr is not None:
            step_size = self.lr
            self._try_update(step_size, params_current, grad_current, closure=closure)
        else:
            batch_step_size = self.state["step_size"]
            step_size = ut.reset_step(step_size=batch_step_size,
                                       n_batches_per_epoch=self.n_batches_per_epoch,
                                       gamma=self.gamma,
                                       reset_option=self.reset_option,
                                       init_step_size=self.init_step_size,
                                       eta_max=1)

            # try taking a Newton step
            dk_list = self._try_update(step_size, params_current, grad_current, closure=closure)

            # verify that the Newton direction results in descent
            phi = compute_phi(grad_current, dk_list)
            assert phi <= 0

            grad_norm = ut.compute_grad_norm(grad_current)

            if grad_norm >= 1e-8:
                found = 0

                for e in range(100):
                    # compute the loss at the next step; no need to compute gradients.
                    loss_next = closure()

                    found, step_size = ssn_line_search(
                                                      step_size=step_size,
                                                      phi=phi,
                                                      loss=loss,
                                                      loss_next=loss_next,
                                                      c=self.c,
                                                      beta=self.beta)
                    if found == 1:
                        break

                    # try a prospective step
                    self._try_update(step_size, params_current, grad_current, closure=closure)

                if found == 0:
                    step_size = 1e-6
                    self._try_update(step_size, params_current, grad_current, closure=closure)

        # save the new step-size
        self.state['step_size'] = step_size
        
        # clear before next round of batch size checking
        self.state['dk'] = None

        return loss

    def _try_update(self, step_size, params_current, grad_current,  closure=None):
        if "dk" in self.state and self.state["dk"] is not None:
            dk = self.state["dk"]
        else:
            dk = hvp.cg_hvp_solve(self.params, grad_current, 
                    closure, alpha=self.lm, max_iter=1000, rtol=1e-6, atol=1e-6, verbose=0,
                    negcurv_flag=1).squeeze()

            self.state["dk"] = dk

        zipped = zip(self.params, params_current, grad_current)
        s_ind = 0
        dk_list = []
        for i, (p, pc, gc) in enumerate(zipped):
            n_params_layer = p.view(-1).shape[0]
            dk_layer = dk[s_ind: s_ind+n_params_layer]
            dk_reshaped = dk_layer.reshape(p.shape)

            p.data = pc - step_size*dk_reshaped
            dk_list += [dk_reshaped]
            s_ind += n_params_layer

        return dk_list


def compute_phi(grad_list, dk_list):
    phi = 0.
    for g, dk in zip(grad_list, dk_list):
        if g is None:
            continue
        phi += g.view(-1).dot(-dk.view(-1))

    return phi


def ssn_line_search(step_size, phi, loss,
                        loss_next, c, beta):
    found = 0

    # computing the new break condition
    break_condition = loss_next - \
        (loss - (step_size) * c * (-phi))

    if (break_condition <= 0):
        found = 1

    else:
        # decrease the step-size by a multiplicative factor
        step_size = step_size * beta

    return found, step_size