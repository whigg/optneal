# genqubo

## install

### install from github repository

```
$ git clone git@github.com:mullzhang/genqubo
$ cd genqubo
$ python setup.py install
```

## How to use

### Python example

```python
import random
import genqubo as gq

N = 10
K = 2
cost_dict = {i: random.gauss(0, 1) for i in range(N)}
cost_mat = gq.dict_to_mat(cost_dict, dims=N)

constraints = [({i: 1 for i in range(N)}, K)]
F, C = gq.const_to_coeff(constraints, dims=N)
cstr_mat, offset = gq.convert_to_penalty(F, C)

lam = 5.0
qubo_mat = cost_mat + lam * cstr_mat
bqm = gq.mat_to_dimod_bqm(qubo_mat, offset)
print(bqm)
```

- [K-hot Problem](examples/ex_khot.py)
- [TSP](examples/ex_tsp.py)
- [Number Partition Problem](examples/ex_num_part.py)