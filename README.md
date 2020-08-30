# Ada-SLS
## Adaptive Gradient Methods Converge Faster with Over-Parameterization (and you can do a line-search) [[Paper]](https://arxiv.org/abs/2006.06835)

Our `AMSGrad Armijo SLS` and `AdaGrad Armijo SLS`  consistently achieve best generalization results.
![](results_sls.png)

## Install requirements
`pip install -r requirements.txt` 


Install the Haven library for managing the experiments with 

```
pip install --upgrade git+https://github.com/ElementAI/haven
```


## Experiments

Run the experiments for the paper using the commands below:

### Synthetic and Kernels

```
python trainval.py -e adaptive_II_syn adaptive_II_kernels -d <datadir> -sb <savedir_base>  -r 1 -c <enable_cuda>
```
where `<datadir>` is where the data is saved (example `.tmp/data`),  `<savedir_base>` is where the results will be saved (example `.tmp/results`), and `<enable_cuda>` is either 0 or 1. It is 1 if the user enables cuda.

### Matrix Factorization Experiments

```
python trainval.py -e adaptive_III_mf -d <datadir> -sb <savedir_base> -r 1
```

### MNIST Experiments

```
python trainval.py -e adaptive_III_mnist -d <datadir> -sb <savedir_base> -r 1
```

### CIFAR Experiments

```
python trainval.py -e adaptive_IV_cifar10 adaptive_IV_cifar100 -d <datadir> -sb <savedir_base> -r 1
```

### ImageNet Experiments

```
python trainval.py -e adaptive_V_imagenet200 adaptive_V_imagenet10 -d <datadir> -sb <savedir_base> -r 1
```


## Results

View results by running the following command.

```
python trainval.py -e adaptive_III_mnist -v 1 -d <datadir> -sb <savedir_base>
```

where `<datadir>` is where the data is saved, and `<savedir_base>` is where the results will be saved.

## Cite
```
@article{vaswani2020adaptive,
  title={Adaptive Gradient Methods Converge Faster with Over-Parameterization (and you can do a line-search)},
  author={Vaswani, Sharan and Kunstner, Frederik and Laradji, Issam and Meng, Si Yi and Schmidt, Mark and Lacoste-Julien, Simon},
  journal={arXiv preprint arXiv:2006.06835},
  year={2020}
}
```
