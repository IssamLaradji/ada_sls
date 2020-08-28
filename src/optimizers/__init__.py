import numpy as np
# from . import sls
from . import others
from . import adaptive_first, adaptive_second
import torch
from src.optimizers import sls, sps, ssn


def get_optimizer(opt, params, n_batches_per_epoch=None, n_train=None, lr=None,
                  train_loader=None, model=None, loss_function=None, exp_dict=None, batch_size=None):
    """
    opt: name or dict
    params: model parameters
    n_batches_per_epoch: b/n
    """
    if isinstance(opt, dict):
        opt_name = opt["name"]
        opt_dict = opt
    else:
        opt_name = opt
        opt_dict = {}

    # ===============================================
    # our optimizers   
    n_batches_per_epoch = opt_dict.get("n_batches_per_epoch") or n_batches_per_epoch    
    

    if opt_name == "adaptive_second":

        opt = adaptive_second.AdaptiveSecond(params,
                    c = opt_dict.get('c', None),
                    n_batches_per_epoch=n_batches_per_epoch,
                    # gv_option=opt_dict.get('gv_option', 'per_param'),
                    base_opt=opt_dict.get('base_opt', 'adagrad'),
                    accum_gv=opt_dict.get('accum_gv', None),
                    lm=opt_dict.get('lm', 0),
                    avg_window=opt_dict.get('window', 10),
                    pp_norm_method=opt_dict.get('pp_norm_method', 'pp_armijo'),
                    momentum=opt_dict.get('momentum', 0),
                    beta=opt_dict.get('beta', 0.99),
                    gamma=opt_dict.get('gamma', 2),
                    # apply_sqrt=opt_dict.get('apply_sqrt', True),
                    init_step_size=opt_dict.get('init_step_size', 1),
                    adapt_flag=opt_dict.get('adapt_flag', 'constant'), 
                    step_size_method=opt_dict.get('step_size_method', 'sls'), 
                    # sls stuff
                    beta_b=opt_dict.get('beta_b', .9),
                    beta_f=opt_dict.get('beta_f', 2.),
                    reset_option=opt_dict.get('reset_option', 1),
                    line_search_fn=opt_dict.get('line_search_fn', "armijo"),   
                    )
                    
    elif opt_name == "adaptive_first":

        opt = adaptive_first.AdaptiveFirst(params,
                    c = opt_dict['c'],
                    n_batches_per_epoch=n_batches_per_epoch,
                    gv_option=opt_dict.get('gv_option', 'per_param'),
                    base_opt=opt_dict['base_opt'],
                    pp_norm_method=opt_dict['pp_norm_method'],
                    momentum=opt_dict.get('momentum', 0),
                    beta=opt_dict.get('beta', 0.99),
                    gamma=opt_dict.get('gamma', 2),
                    init_step_size=opt_dict.get('init_step_size', 1),
                    adapt_flag=opt_dict.get('adapt_flag', 'constant'), 
                    step_size_method=opt_dict['step_size_method'], 
                    # sls stuff
                    beta_b=opt_dict.get('beta_b', .9),
                    beta_f=opt_dict.get('beta_f', 2.),
                    reset_option=opt_dict.get('reset_option', 1),
                    line_search_fn=opt_dict.get('line_search_fn', "armijo"),   
                    )
   
    elif opt_name == "sgd_armijo":
        # if opt_dict.get("infer_c"):
        #     c = (1e-3) * np.sqrt(n_batches_per_epoch)
        if opt_dict['c'] == 'theory':
            c = (n_train - batch_size) / (2 * batch_size * (n_train - 1))
        else:
            c = opt_dict.get("c") or 0.1
        
        opt = sls.Sls(params,
                    c = c,
                    n_batches_per_epoch=n_batches_per_epoch,
                    init_step_size=opt_dict.get("init_step_size", 1),
                    line_search_fn=opt_dict.get("line_search_fn", "armijo"), 
                    gamma=opt_dict.get("gamma", 2.0),
                    reset_option=opt_dict.get("reset_option", 1),
                    eta_max=opt_dict.get("eta_max"))


    elif opt_name == "sgd_goldstein":
        opt = sls.Sls(params, 
                      c=opt_dict.get("c") or 0.1,
                      reset_option=opt_dict.get("reset_option") or 0,
                      n_batches_per_epoch=n_batches_per_epoch,
                      line_search_fn="goldstein")

    elif opt_name == "sgd_nesterov":
        opt = sls.SlsAcc(params, 
                        acceleration_method="nesterov", 
                        gamma=opt_dict.get("gamma", 2.0),
                        aistats_eta_bound=opt_dict.get("aistats_eta_bound", 10.0))

    elif opt_name == "sgd_polyak":
        opt = sls.SlsAcc(params, 
                         c=opt_dict.get("c") or 0.1,
                         momentum=opt_dict.get("momentum", 0.6),
                         n_batches_per_epoch=n_batches_per_epoch,
                         gamma=opt_dict.get("gamma", 2.0),
                         acceleration_method="polyak",
                         aistats_eta_bound=opt_dict.get("aistats_eta_bound", 10.0),
                         reset_option=opt_dict.get("reset", 0))

    elif opt_name == "seg":
        opt = sls.SlsEg(params, n_batches_per_epoch=n_batches_per_epoch)

    elif opt_name == "ssn":
        opt = ssn.Ssn(params, 
            n_batches_per_epoch=n_batches_per_epoch, 
            init_step_size=opt_dict.get('init_step_size', 1.0), 
            lr=None, 
            c=opt_dict.get('c',0.1), 
            beta=0.9, 
            gamma=1.5,
            reset_option=1, 
            lm=opt_dict.get("lm", 0))

    # ===============================================
    # others
    elif opt_name == "adam":
        opt = torch.optim.Adam(params, amsgrad=opt.get('amsgrad'),  lr=opt['lr'],  betas=opt.get('betas', (0.9,0.99)))

    elif opt_name == "adagrad":
        opt = torch.optim.Adagrad(params, lr=opt['lr'])

    elif opt_name == 'sgd':
        # best_lr = lr if lr else 1e-3
        opt = torch.optim.SGD(params, lr=opt['lr'])

    elif opt_name == "sgd-m":
        best_lr = lr if lr else 1e-3
        opt = torch.optim.SGD(params, lr=best_lr, momentum=0.9)

    elif opt_name == 'rmsprop':
        opt = torch.optim.RMSprop(params, lr=opt['lr'])

    elif opt_name == 'adabound':
        opt = others.AdaBound(params)
        print('Running AdaBound..')

    elif opt_name == 'amsbound':
        opt = others.AdaBound(params, amsbound=True)

    elif opt_name == 'sps':
        opt = sps.Sps(params, c=opt_dict["c"], 
                        n_batches_per_epoch=n_batches_per_epoch, 
                        adapt_flag=opt_dict.get('adapt_flag', 'basic'),
                        fstar_flag=opt_dict.get('fstar_flag'),
                        eta_max=opt_dict.get('eta_max'),
                        eps=opt_dict.get('eps', 0))


    elif opt_name == 'lookahead':
        base_opt = torch.optim.Adam(params, lr=1e-3, betas=(0.9, 0.999)) # Any optimizer
        opt = others.Lookahead(base_opt, k=5, alpha=0.5) # Initialize Lookahead

        # base_opt = torch.optim.Adam(params)
        # opt = others.Lookahead(base_opt)

    elif opt_name == 'radam':
        opt = others.RAdam(params)

    elif opt_name == 'plain_radam':
        opt = others.PlainRAdam(params)


    else:
        raise ValueError("opt %s does not exist..." % opt_name)

    return opt




