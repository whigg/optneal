import functools


class BaseOptimizer:
    def __init__(self, child_sampler, sampler_params=None):
        self.child_sampler = child_sampler
        self.func_sample = functools.partial(self.child_sampler.sample, **sampler_params)
        self.sampler_params = sampler_params

    is_conv = False

    def optimize(self):
        pass
