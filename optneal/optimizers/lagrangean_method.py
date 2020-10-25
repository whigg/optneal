from .base_optimizer import BaseOptimizer

import optneal as opn


class LagrangeRelax(BaseOptimizer):
    def optimize(self, cost, lagrange, num_iters, **params):
        multp = lagrange.multp
        lagrange_ = lagrange.copy()

        sampleset_hists = []
        for _ in range(num_iters):
            cost_func = cost + lagrange_
            bqm = cost_func.to_dimod_bqm()

            sampleset = self.func_sample(bqm)
            sampleset_hists.append(sampleset)

            multp_update = self._update_multp(multp, sampleset)
            lagrange_.change_multp(multp_update)

            if self.is_conv:
                break

        return sampleset_hists

    def _update_multp(self, multp, sampleset):
        multp_update = multp
        return multp_update


class AugumentedLagrange(BaseOptimizer):
    def optimize(self, cost, lagrange, num_iters, **params):
        multp = lagrange.multp
        lagrange_ = lagrange.copy()

        sampleset_hists = []
        for _ in range(num_iters):
            cost_func = cost + lagrange_
            bqm = cost_func.to_dimod_bqm()

            sampleset = self.func_sample(bqm)
            sampleset_hists.append(sampleset)

            multp_update = self._update_multp(multp, sampleset)
            lagrange_.change_multp(multp_update)

            if self.is_conv:
                break

        return sampleset_hists
