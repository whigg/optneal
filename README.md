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
import genqubo as gq

N = 2
constraints = [({(i, j): 1 for j in range(N)}, 1) for i in range(N)]
constraints += [({(i, j): 1 for i in range(N)}, 1) for j in range(N)]
F, C = gq.const_coeff(constraints, dims=(N, N))
qubo_mat, offset = gq.convert_to_penalty(F, C)

bqm = gq.mat_to_dimod_bqm(qubo_mat, offset)
print(bqm)
```