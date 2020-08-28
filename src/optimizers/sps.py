import torch
import copy
import time
import numpy as np

from src.optimizers.base import utils as ut

class Sps(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 init_step_size=1,
                 c=0.1,
                 gamma=2.0,
                 eta_max=10,
                 adapt_flag=None,
                 fstar_flag=None,
                 eps=0):
        params = list(params)
        super().__init__(params, {})
        self.eps = eps
        self.params = params
        self.c = c
        self.eta_max = eta_max
        self.gamma = gamma
        self.init_step_size = init_step_size
        self.adapt_flag = adapt_flag
        self.state['step'] = 0
        self.state['step_size_avg'] = 0.

        self.state['step_size'] = init_step_size
        self.step_size_max = 0.
        self.n_batches_per_epoch = n_batches_per_epoch

        self.state['n_forwards'] = 0
        self.state['n_backwards'] = 0
        self.state['lb'] = None
        self.loss_min = np.inf
        self.loss_sum = 0.
        self.loss_max = 0.
        self.fstar_flag = fstar_flag

    def step(self, closure, batch, clip_grad=False):
        indices = batch['meta']['indices']
        # deterministic closure
        seed = time.time()

        batch_step_size = self.state['step_size']

        if self.fstar_flag:
            fstar = float(batch['meta']['fstar'].mean())
        else:
            fstar = 0.

        # get loss and compute gradients
        loss = closure()
        loss.backward()

        if clip_grad:
            torch.nn.utils.clip_grad_norm_(self.params, 0.25)

        # increment # forward-backward calls
        self.state['n_forwards'] += 1
        self.state['n_backwards'] += 1

        if self.state['step'] % int(self.n_batches_per_epoch) == 0:
            self.state['step_size_avg'] = 0.

        self.state['step'] += 1

        # save the current parameters:
        params_current = copy.deepcopy(self.params)
        grad_current = ut.get_grad_list(self.params)

        grad_norm = ut.compute_grad_norm(grad_current)
        
        if grad_norm < 1e-6:
            return 0.

        if self.adapt_flag == 'moving_lb':
            if (self.state['lb'] is not None) and loss > self.state['lb']: 
                step_size = (loss - self.state['lb']) / (self.c * (grad_norm)**2)
                step_size = step_size.item()
                loss_scalar = loss.item()
            else:
                step_size = 0.
                loss_scalar = loss.item()

            if (self.state['step'] % int(self.n_batches_per_epoch)) == 0:
                # do update on lower bound 
                # new_bound = self.loss_sum / self.n_batches_per_epoch
                # self.state['lb'] = (self.loss_sum / self.n_batches_per_epoch) / 100.
                # self.state['lb'] /= 2.
                # self.state['lb'] = self.loss_max / 1000.
                self.state['lb'] = self.loss_min / 100.
                # if new_bound > self.state['lb']:
                #     self.state['lb'] = (new_bound - self.state['lb']) / 2.
                # self.loss_sum = 0.
                print('lower_bound:', self.state['lb'])

            self.loss_sum += loss_scalar
            self.loss_min = min(loss_scalar, self.loss_min)
            self.loss_max = max(loss_scalar, self.loss_max)

        elif self.adapt_flag in ['constant']:
            step_size = (loss - fstar) / (self.c * (grad_norm)**2 + self.eps)
            if loss < fstar:
                step_size = 0.
                loss_scalar = 0.
            else:
                if self.eta_max is None:
                    step_size = step_size.item()
                else:
                    step_size =  min(self.eta_max, step_size.item())
                    
                loss_scalar = loss.item()

        elif self.adapt_flag in ['basic']:
            if torch.isnan(loss):
                raise ValueError('loss is NaN')
            # assert(loss >= fstar)
            if loss < fstar:
                step_size = 0.
                loss_scalar = 0.
            else:
                step_size = (loss - fstar) / (self.c * (grad_norm)**2+ self.eps)
                step_size =  step_size.item()

                loss_scalar = loss.item()

        elif self.adapt_flag in ['smooth_iter']:
            step_size = loss / (self.c * (grad_norm)**2)
            coeff = self.gamma**(1./self.n_batches_per_epoch)
            step_size =  min(coeff * self.state['step_size'], 
                             step_size.item())
           
            loss_scalar = loss.item()

        elif self.adapt_flag == 'smooth_epoch':
            step_size = loss / (self.c * (grad_norm)**2)
            step_size = step_size.item()
            if self.state['step_size_epoch'] != 0:
                step_size =  min(self.state['step_size_epoch'], 
                                 step_size)
                self.step_size_max = max(self.step_size_max, step_size)
            else:
                self.step_size_max = max(self.step_size_max, step_size)
                step_size = 0.
                

            loss_scalar = loss.item()

             # epoch done
            if (self.state['step'] % int(self.n_batches_per_epoch)) == 0:
                self.state['step_size_epoch'] = self.step_size_max
                self.step_size_max = 0.


        # update
        # zipped = zip(self.params, params_current, grad_current)

        # for p_next, p_current, g_current in zipped:
        #     if g_current is None:
        #         continue
        #     p_next.data[:] = p_current.data
        #     p_next.data.add_(- step_size, g_current)

        # for p in self.params:
        #     if p.grad is None:
        #         continue
        #     d_p = p.grad.data
        #     p.data.add_(-float(step_size), d_p)

            # p.data = p.data - step_size * p.grad.data
        ut.try_sgd_update(self.params, step_size, params_current, grad_current)

        # save the new step-size
        self.state['step_size'] = step_size

        
        self.state['step_size_avg'] += (step_size / self.n_batches_per_epoch)
        self.state['grad_norm'] = grad_norm.item()
        
        if torch.isnan(self.params[0]).sum() > 0:
            print('nan')

        return loss
