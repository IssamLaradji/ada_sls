import torch
import copy
import time
import math
import warnings
import torch
import numpy as np

from backpack import backpack
from backpack.extensions import DiagHessian, DiagGGNExact, DiagGGNMC


from src.optimizers.base import utils as ut

import torch
import copy
import time
import math
import warnings
import torch
import numpy as np
from . import adaptive_first
from src.optimizers.base import utils as ut

class AdaptiveSecond(adaptive_first.AdaptiveFirst):
    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 init_step_size=1,
                 c=0.1,
                 gamma=2.0,
                 eta_max=None,
                 beta=0.99,
                 momentum=None,

                 # these are the preconditioner specific options
                 base_opt='diag_hessian',
                 accum_gv=None,
                 lm=0,
                 avg_window=10,
                 pp_norm_method='pp_armijo',
                 step_size_method='sls',
                 
                 # sps stuff
                 adapt_flag='constant',

                 # sls stuff
                 beta_b=0.9,
                 beta_f=2.0,
                 reset_option=1,
                 line_search_fn="armijo",
                 ):
        params = list(params)
        super().__init__(params, {})

        # splr stuff
        self.adapt_flag = adapt_flag 

        # sls stuff
        self.beta_f = beta_f
        self.beta_b = beta_b
        self.reset_option = reset_option
        self.line_search_fn = line_search_fn

        # others
        self.params = params
        self.c = c
        self.eta_max = eta_max
        self.gamma = gamma
        self.momentum = momentum
        self.init_step_size = init_step_size
        self.state['step'] = 0
        self.state['step_size_avg'] = 0.
        self.beta = beta
        self.state['step_size'] = init_step_size
        self.step_size_max = 0.
        self.n_batches_per_epoch = n_batches_per_epoch

        self.base_opt = base_opt
        self.state['gv'] = None
        self.accum_gv = accum_gv
        self.lm = lm
        self.avg_window = avg_window
        self.pp_norm_method = pp_norm_method

        self.state['n_forwards'] = 0
        self.state['n_backwards'] = 0

        self.step_size_method = step_size_method

        # gv options
        self.state['step'] = 0

    def step(self, closure, clip_grad=False):
        # increment step
        self.state['step'] += 1

        # deterministic closure
        seed = time.time()
        def closure_deterministic(for_backtracking=False):
            with ut.random_seed_torch(int(seed)):
                return closure(for_backtracking)
        
        # get loss and compute gradients/second-order extensions
        loss = closure_deterministic()
        if self.base_opt == 'diag_hessian':
            with backpack(DiagHessian()):            
                loss.backward()

        elif self.base_opt == 'diag_ggn_ex':
            with backpack(DiagGGNExact()):
                loss.backward()

        elif self.base_opt == 'diag_ggn_mc':
            with backpack(DiagGGNMC()):
                loss.backward()
        else:
            loss.backward()

        if clip_grad:
            torch.nn.utils.clip_grad_norm_(self.params, 0.25)
        # increment # forward-backward calls
        self.state['n_forwards'] += 1
        self.state['n_backwards'] += 1        
        # save the current parameters:
        params_current = copy.deepcopy(self.params)
        grad_current = ut.get_grad_list(self.params)
        grad_norm = ut.compute_grad_norm(grad_current)

        # keep track of step
        if self.state['step'] % int(self.n_batches_per_epoch) == 1:
            self.state['step_size_avg'] = 0.

        # if grad_norm < 1e-6:
        #     return 0.

        #  Gv options
        # =============
        # update gv
        if self.base_opt == 'diag_hessian':
            # get diagonal hessian here and store it in state['gv']
            gv = [p.diag_h for p in self.params]

        elif self.base_opt == 'diag_ggn_ex':
            gv = [p.diag_ggn_exact for p in self.params]

        elif self.base_opt == 'diag_ggn_mc':
            gv = [p.diag_ggn_mc for p in self.params]

        else:
            raise ValueError('%s does not exist' % self.gv_update)

        for gv_i in gv:
            if torch.any(gv_i < 0):
                warnings.warn("%s contains negative values." % (self.gv_update))
                print(gv)

        if self.state['gv'] is None or self.accum_gv is None:
            self.state['gv'] = gv

            if self.accum_gv == 'avg':
                # first iteration
                self.state['gv_lag'] = [[gv_i] for gv_i in gv] 
                self.state['gv_sum'] = gv

        elif self.accum_gv == 'max':
            for i, (gv_old, gv_new) in enumerate(zip(self.state['gv'], gv)):
                self.state['gv'][i] = torch.max(gv_old, gv_new)

        elif self.accum_gv == 'sum':
            for i, (gv_old, gv_new) in enumerate(zip(self.state['gv'], gv)):
                self.state['gv'][i] = gv_old + gv_new   

        elif self.accum_gv == 'ams':
            for i, (gv_old, gv_new) in enumerate(zip(self.state['gv'], gv)):
                gv_accum = self.beta*gv_old + (1-self.beta)*gv_new
                self.state['gv'][i] = torch.max(gv_old, gv_accum)

        elif self.accum_gv == 'ams_no_max':
            for i, (gv_old, gv_new) in enumerate(zip(self.state['gv'], gv)):
                # same as above without the max
                gv_accum = self.beta*gv_old + (1-self.beta)*gv_new
                self.state['gv'][i] = gv_accum

        elif self.accum_gv == 'avg':
            t = self.state['step']
            
            for i, (gv_new, gv_lag) in enumerate(zip(gv, self.state['gv_lag'])):

                if t < self.avg_window:
                    # keep track of the sum only, no need to kick anyone out 
                    # for the first window number of iterations
                    gv_sum = self.state['gv_sum'][i] + gv_new
                    # take the average of all seen so far
                    gv_accum = gv_sum / t 

                else:
                    # kick out the gv stored for iteration (t-window)
                    gv_kick_out = gv_lag.pop(0)
                    # update the sum for the past window iterations
                    gv_sum = self.state['gv_sum'][i] + gv_new - gv_kick_out
                    # take the average gv within the window
                    gv_accum = gv_sum / self.avg_window

                # add the new gv to the lags and update sum
                self.state['gv_lag'][i].append(gv_new)
                self.state['sum'][i] = gv_sum
                # this will be used as the diagonal preconditioner
                self.state['gv'][i] = gv_accum

        else:
            raise ValueError('accum_gv %s does not exist' % self.accum_gv)

        if self.lm > 0:
            for i in range(len(self.state['gv'])):
                self.state['gv'][i] += self.lm 

        # compute pp norm method
        pp_norm = self.get_pp_norm(grad_current=grad_current)
        
        # compute step size - same as the SLS code but with a different norm pp_norm
        # =================
        step_size = self.get_step_size(closure_deterministic, loss, params_current, grad_current, grad_norm, pp_norm, 
                        for_backtracking=True)
            

        self.try_sgd_precond_update(self.params, step_size, params_current,
                            grad_current)

        # save the new step-size
        self.state['step_size'] = step_size

        self.state['step_size_avg'] += (step_size / int(self.n_batches_per_epoch))
        self.state['grad_norm'] = grad_norm.item()
  
        if torch.isnan(self.params[0]).sum() > 0:
            print('nan')

        return loss