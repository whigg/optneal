# Optneal

Optneal is a Python module for mathematical optimization using annealing and LP solvers.

- Typical transformation of cost function and constraints to unconstrained form.
- Generic optimization method or algorithm, such as penalty method or lagrange multiplier method.
- Parameter tuning (*TBI)

## install

### install from github repository

```
$ git clone git@github.com:mullzhang/optneal
$ cd optneal
$ python setup.py install
```

### install with pip command

```
$ pip install git+https://github.com/mullzhang/optneal.git
```

## How to use

### Python example

```python
import random
import optneal as opn

N = 10
K = 2
numbers = [random.uniform(0, 5) for _ in range(N)]
cost_dict = {i: numbers[i] for i in range(N)}
cost = opn.Cost(cost_dict, shape=N)

constraints = [({i: 1 for i in range(N)}, K)]
penalty = opn.Penalty(constraints, shape=N)

lam = 5.0
cost_func = cost + lam * penalty.normalize()
bqm = cost_func.to_dimod_bqm()
print(bqm)
```

### Typical problems

- [K-hot Problem](examples/ex_khot.py)
- [TSP](examples/ex_tsp.py)
- [Number Partition Problem](examples/ex_num_part.py)

## Benchmark

### Generating model

TSP: [Source code](examples/benchmark.py)

![benchmark](https://github.com/mullzhang/optneal/blob/master/examples/elapsed_time.png)
