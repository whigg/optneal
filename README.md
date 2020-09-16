# genqubo

## install

### install from github repository

```
$ git clone git@github.com:maruzhang/par-pyqubo
$ cd par_pyqubo
$ python setup.py install
```

## How to use

### Python example

```python
import genqubo as gq

N = 2
constraints = [({(i, j): 1 for j in range(N)}, 1) for i in range(N)]
constraints += [({(i, j): 1 for i in range(N)}, 1) for j in range(N)]
F, C = gq.const_coeff(constraints, dims=(N, N))
qubo_mat, offset = gq.convert_to_penalty(F, C)
```