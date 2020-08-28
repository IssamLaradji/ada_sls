from haven import haven_utils as hu
import itertools
# RUNS = [0, 1]
# RUNS = [0,1,2,3,4]
RUNS = [0, 1, 2, 3, 4]


def get_benchmark(benchmark, opt_list):
    if benchmark == 'syn':
        return {"dataset": ["synthetic"],
                "model": ["logistic"],
                "loss_func": ["softmax_loss"],
                "opt": opt_list,
                "acc_func": ["softmax_accuracy"],
                'margin':
                [
                    0.05,
            0.1,
                    0.5,
                    0.01,
        ],
            "n_samples": [1000],
            "d": 20,
            "batch_size": [100],
            "max_epoch": [200],
            "runs": RUNS}

    elif benchmark == 'kernels':
        return {"dataset": ["mushrooms", "ijcnn", "rcv1"],
                "model": ["logistic"],
                "loss_func": ['softmax_loss'],
                "acc_func": ["softmax_accuracy"],
                "opt": opt_list,
                "batch_size": [100],
                "max_epoch": [100],
                "runs": RUNS}

    elif benchmark == 'mf':
        return {"dataset": ["matrix_fac"],
                "model": ["matrix_fac_1", "matrix_fac_4", "matrix_fac_10", "linear_fac"],
                "loss_func": ["squared_loss"],
                "opt": opt_list,
                "acc_func": ["mse"],
                "batch_size": [100],
                "max_epoch": [100],
                "runs": RUNS}

    elif benchmark == 'mnist':
        return {"dataset": ["mnist"],
                "model": ["mlp"],
                "loss_func": ["softmax_loss"],
                "opt": opt_list,
                "acc_func": ["softmax_accuracy"],
                "batch_size": [128],
                "max_epoch": [100],
                "runs": RUNS}

    elif benchmark == 'cifar10':
        return {"dataset": ["cifar10"],
                "model": [
            "densenet121",

            "resnet34"
        ],
            "loss_func": ["softmax_loss"],
            "opt": opt_list,
            "acc_func": ["softmax_accuracy"],
            "batch_size": [128],
            "max_epoch": [200],
            "runs": RUNS}

    elif benchmark == 'cifar100':
        return {"dataset": ["cifar100"],
                "model": [
            "densenet121_100",
            "resnet34_100"
        ],
            "loss_func": ["softmax_loss"],
            "opt": opt_list,
            "acc_func": ["softmax_accuracy"],
            "batch_size": [128],
            "max_epoch": [200],
            "runs": RUNS}

    elif benchmark == 'cifar10_nobn':
        return {"dataset": ["cifar10"],
                "model": ["resnet34_nobn", "densenet121_nobn"],
                "loss_func": ["softmax_loss"],
                "opt": opt_list,
                "acc_func": ["softmax_accuracy"],
                "batch_size": [128],
                "max_epoch": [200],
                "runs": RUNS}

    elif benchmark == 'cifar100_nobn':
        return {"dataset": ["cifar100"],
                "model": ["resnet34_100_nobn", "densenet121_100_nobn"],
                "loss_func": ["softmax_loss"],
                "opt": opt_list,
                "acc_func": ["softmax_accuracy"],
                "batch_size": [128],
                "max_epoch": [200],
                "runs": RUNS}

    elif benchmark == 'imagenet200':
        return {"dataset": ["tiny_imagenet"],
                "model": ["resnet18"],
                "loss_func": ["softmax_loss"],
                "opt": opt_list,
                "acc_func": ["softmax_accuracy"],
                "batch_size": [128],
                "max_epoch": [200],
                "runs": RUNS}
    elif benchmark == 'imagenet10':
        return {"dataset": ["imagenette2-160", "imagewoof2-160"],
                "model": ["resnet18"],
                "loss_func": ["softmax_loss"],
                "opt": opt_list,
                "acc_func": ["softmax_accuracy"],
                "batch_size": [128],
                "max_epoch": [100],
                "runs": RUNS}


EXP_GROUPS = {}
# -------------- ## -------------- ## -------------- ## -------------- #
# Setting up optimizers
# ------------------ #
# I. optimizers with best constant step-size
# 1. Adam including momentum
adam_constant_list = []
adam_constant_list += [
    {'name': 'adam', 'lr': lr, 'betas': [0, 0.99]} for lr in [1000, 500, 100, 50, 10., 1, 1e-1, 1e-2, 1e-3]
]
adam_constant_list += [
    {'name': 'adam', 'lr': lr, 'betas': [0.9, 0.99]} for lr in [1000, 500, 100, 50, 10., 1, 1e-1, 1e-2, 1e-3]
]


# 2. AMSgrad
amsgrad_constant_list = []
amsgrad_constant_list += [
    {'name': 'adam', 'lr': lr, 'betas': [0, 0.99], 'amsgrad':True} for lr in [1000, 500, 100, 50, 10., 1, 1e-1, 1e-2, 1e-3]
]
amsgrad_constant_list += [
    {'name': 'adam', 'lr': lr, 'betas': [0.9, 0.99], 'amsgrad':True} for lr in [1000, 500, 100, 50, 10., 1, 1e-1, 1e-2, 1e-3]
]


# 3. Adagrad
adagrad_constant_list = []
adagrad_constant_list += [
    {'name': 'adagrad', 'lr': lr} for lr in [1000, 500, 100, 50, 10., 1, 1e-1, 1e-2, 1e-3]
]

constant_list = adam_constant_list + amsgrad_constant_list + adagrad_constant_list

# ------------------ #
# II. SGD with SLS  + SPS
# 1. SLS
sls_list = []
c_list = [.1, .2, 0.5]
for c in c_list:
    sls_list += [{'name': "sgd_armijo", 'c': c, 'reset_option': 1}]

# 2. SPS
sps_list = []
c_list = [.2, .5, 1.0]
for c in c_list:
    sps_list += [{'name': "sps", 'c': c, 'adapt_flag': 'smooth_iter'}]

sgd_list = sls_list + sps_list

# ------------------ #
# III. Adaptive with first order preconditioners + line-search/SPS
# 1. Adaptive + SLS
# 1.1. Lipschitz + Adagrad
reset_option_list = [0, 1]
c_list = [0.1, 0.5, 0.75]

adaptive_first_sls_lipschitz_list = []
for c in c_list:
    for reset_option in reset_option_list:
        adaptive_first_sls_lipschitz_list += [{'name': 'adaptive_first',
                                               'c': c,
                                               'gv_option': 'per_param',
                                               'base_opt': 'adagrad',
                                               'pp_norm_method': 'pp_lipschitz',
                                               'init_step_size': 100,  # setting init step-size to 100. SLS should be robust to this
                                               "momentum": 0.,
                                               'step_size_method': 'sls',
                                               'reset_option': reset_option}]

# 1.2. Armijo + Adam / Amsgrad / Amsgrad
adaptive_first_sls_armijo_list = []
reset_option_list = [0, 1]
c_list = [0.1, 0.2, 0.5]
base_opt_list = ['adam', 'amsgrad', 'adagrad']

for base_opt in base_opt_list:
    for c in c_list:
        for reset_option in reset_option_list:
            adaptive_first_sls_armijo_list += [{'name': 'adaptive_first',
                                                        'c': c,
                                                        'gv_option': 'per_param',
                                                        'base_opt': base_opt,
                                                        'pp_norm_method': 'pp_armijo',
                                                        'init_step_size': 100,  # setting init step-size to 100. SLS should be robust to this
                                                        "momentum": 0.,
                                                        'step_size_method': 'sls',
                                                'reset_option': reset_option}]

adaptive_first_sls_list = adaptive_first_sls_lipschitz_list + \
    adaptive_first_sls_armijo_list

# 2. Adaptive + SPS / Only Armijo
c_list = [0.2, 0.5, 1.0]
base_opt_list = ['adam', 'amsgrad', 'adagrad']

adaptive_first_sps_list = []
for base_opt in base_opt_list:
    for c in c_list:
        adaptive_first_sps_list += [{'name': 'adaptive_first',
                                     'c': c,
                                     'gv_option': 'per_param',
                                     'base_opt': base_opt,
                                     'pp_norm_method': 'pp_armijo',
                                     'init_step_size': 1,
                                     "momentum": 0.,
                                     'step_size_method': 'sps',
                                     'adapt_flag': 'smooth_iter'}]

adaptive_first_list = adaptive_first_sls_list + adaptive_first_sps_list

# ------------------ #
# IV. Adaptive with second order preconditioners + line-search/SPS
# 1. SLS + Armijo
adaptive_second_sls_list = []
c_list = [0.2, 0.5, 1.0]
reset_option_list = [0, 1]

for c in c_list:
    for reset_option in reset_option_list:
        adaptive_second_sls_list += [{'name': 'adaptive_second',
                                      'c': c,
                                      'gv_option': 'per_param',
                                      'base_opt': 'diag_ggn_ex',
                                      'pp_norm_method': 'pp_armijo',
                                      'init_step_size': 1,
                                      "momentum": 0.,
                                      'backpack': True,
                                      'step_size_method': 'sls',
                                      'reset_option': reset_option}]

# 2. SPS + Armijo
adaptive_second_sps_list = []
c_list = [0.2, 0.5, 1.0]
for c in c_list:
    adaptive_second_sps_list += [{'name': 'adaptive_second',
                                  'c': c,
                                  'gv_option': 'per_param',
                                  'base_opt': 'diag_ggn_ex',
                                  'pp_norm_method': 'pp_armijo',
                                  'init_step_size': 1,
                                  "momentum": 0.,
                                  'backpack': True,
                                  'step_size_method': 'sps',
                                  'adapt_flag': 'smooth_iter'}]

adaptive_second_list = adaptive_second_sls_list + adaptive_second_sps_list


baselines_list = [{"name": "adabound"}, {
    "name": "radam"}, {"name": "plain_radam"}, ]
# -------------- ## -------------- ## -------------- ## -------------- #
# Setting up benchmarks
# ------------------ #

# ------------------ #
# II. Convex with interpolation
benchmarks_list = ['syn', 'kernels']
# all optimizers for small exps
opt_list = adaptive_second_list + adaptive_first_list + baselines_list + constant_list + sgd_list 
           
for benchmark in benchmarks_list:
    EXP_GROUPS['adaptive_II_%s' % benchmark] = hu.cartesian_exp_group(
        get_benchmark(benchmark, opt_list))

# ------------------ #
# III. Easy nonconvex
benchmarks_list = ['mnist', 'mf']
opt_list = adaptive_first_sls_lipschitz_list + amsgrad_constant_list + adam_constant_list + \
    baselines_list + sgd_list + adaptive_first_sls_armijo_list + adaptive_first_sps_list
for benchmark in benchmarks_list:
    EXP_GROUPS['adaptive_III_%s' % benchmark] = hu.cartesian_exp_group(
        get_benchmark(benchmark, opt_list))

# ------------------ #
# IV. Larg-scale nonconvex
benchmarks_list = ['cifar10_nobn', 'cifar100_nobn', 'cifar10', 'cifar100']
opt_list = adaptive_first_sls_lipschitz_list + adaptive_first_sls_armijo_list + \
    baselines_list + adam_constant_list + \
    amsgrad_constant_list + sgd_list + adaptive_first_sps_list
for benchmark in benchmarks_list:
    EXP_GROUPS['adaptive_IV_%s' % benchmark] = hu.cartesian_exp_group(
        get_benchmark(benchmark, opt_list))

# ------------------ #
# V. Larg-scale nonconvex
benchmarks_list = ['imagenet200', 'imagenet10']

imagenet_opt_list = [
    {'name': 'adaptive_first', 'c': 0.1, 'gv_option': 'per_param',
        'base_opt': 'amsgrad', 'pp_norm_method': 'pp_armijo',
        'init_step_size': 100, 'momentum': 0.0,
        'step_size_method': 'sls', 'reset_option': 1},

    {'name': 'adaptive_first', 'c': 0.1, 'gv_option': 'per_param',
        'base_opt': 'adagrad', 'pp_norm_method': 'pp_armijo',
        'init_step_size': 100, 'momentum': 0.0, 'step_size_method': 'sls',
        'reset_option': 1},


    {'name': 'adabound'},

    {'name': 'radam'},

    {'name': 'plain_radam'},

    {'name': 'adam', 'lr': 0.001, 'betas': [0.9, 0.99]},

    {'name': 'sgd_armijo', 'c': 0.1, 'reset_option': 1},

    {'name': 'sps', 'c': 0.2, 'adapt_flag': 'smooth_iter'},

    {'name': 'adaptive_first', 'c': 0.5,
     'gv_option': 'per_param', 'base_opt': 'adam', 'pp_norm_method': 'pp_armijo',
     'init_step_size': 1, 'momentum': 0.0, 'step_size_method': 'sps',
     'adapt_flag': 'smooth_iter'}, 
     
     ]


for benchmark in benchmarks_list:
    EXP_GROUPS['adaptive_V_%s' % benchmark] = hu.cartesian_exp_group(
        get_benchmark(benchmark, imagenet_opt_list))





### =========== second order ========== ###
opt_list = []

accum_gv_list = [None, 'ams', 'ams_no_max', 'max', 'sum']

# SPS
c_list_sps = [.2, .5, 1.0]
for c in c_list_sps:
    for accum_gv in accum_gv_list:
        opt_list.append({'name':'adaptive_second', 
                'c':c, 
                'gv_option':'per_param',
                'base_opt':'diag_ggn_ex',
                'accum_gv':accum_gv,
                'pp_norm':'pp_armijo',
                'init_step_size':1,
                "momentum":0.,
                'backpack':True,
                'step_size_method':'sps',
                'adapt_flag':'smooth_iter'}),

# SLS
c_list_sls = [.1, .2, .5]
for c in c_list_sls:
    for accum_gv in accum_gv_list:
        opt_list.append({'name':'adaptive_second', 
                    'c':c, 
                    'gv_option':'per_param',
                    'base_opt':'diag_ggn_ex',
                    'accum_gv':accum_gv,
                    'pp_norm':'pp_armijo',
                    'init_step_size':100,
                    "momentum":0.,
                    'backpack':True,
                    'step_size_method':'sls',
                    'reset_option': 1})

diag_hessian_list = []
for c in [0.5, 1.0]:
    # SLS
    diag_hessian_list.append({'name':'adaptive_second', 
                    'c':c, 
                    'gv_option':'per_param',
                    'base_opt':'diag_ggn_ex',
                    'accum_gv':None,
                    'lm':1e-3,
                    'pp_norm':'pp_armijo',
                    'init_step_size':100,
                    "momentum":0.,
                    'backpack':True,
                    'step_size_method':'sls',
                    'reset_option': 1})

    # SPS
    diag_hessian_list.append({'name':'adaptive_second', 
                'c':c, 
                'gv_option':'per_param',
                'base_opt':'diag_ggn_ex',
                'accum_gv':None,
                'lm':1e-3,
                'pp_norm':'pp_armijo',
                'init_step_size':1,
                "momentum":0.,
                'backpack':True,
                'step_size_method':'sps',
                'adapt_flag':'smooth_iter'})

opt_list += diag_hessian_list


ssn_list = []
for c in c_list_sls:
    ssn_list.append({'name':'ssn',
              'lm':1e-3,
              'c':c,
              'init_step_size':100})

opt_list += ssn_list


EXP_GROUPS['adaptive_II_syn'] += hu.cartesian_exp_group({"dataset":["synthetic"],
                                            "model":["logistic"],
                                            "loss_func": ["softmax_loss"],
                                            "opt": opt_list,
                                            "acc_func":["softmax_accuracy"],
                                            'margin':[ 0.05, 0.1, 0.5,  0.01,],
                                            # "scale": 100,
                                            "n_samples": [1000],
                                            "d": 20,
                                            "batch_size":[100],
                                            "max_epoch":[200],
                                            "runs":RUNS})


EXP_GROUPS['adaptive_II_kernels'] += hu.cartesian_exp_group({"dataset":["mushrooms", 'ijcnn', 'rcv1'],
                                            "model":["logistic"],
                                            "loss_func": ["softmax_loss"],
                                            "opt": opt_list,
                                            "acc_func":["softmax_accuracy"],
                                            "batch_size":[100],
                                            "max_epoch":[100],
                                            "runs":RUNS})