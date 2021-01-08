import random
import numpy as np
import dimod
from deap import base, tools, creator


class GASampler(dimod.Sampler):
    """ Genetic Algorithm Sampler """

    def __init__(self):
        self._properties = {}
        self._parameters = {}

    @property
    def properties(self):
        return self._properties

    @property
    def parameters(self):
        return self._parameters

    def _calc_ene(self, individual, bqm_list):
        return [bqm.energy(individual) for bqm in bqm_list]

    def _apply_fit(self, individuals):
        fitnesses = map(self.toolbox.evaluate, individuals)
        for ind, fit in zip(individuals, fitnesses):
            ind.fitness.values = fit
        return individuals

    def _regist_toolbox(self, bqm_list):
        self.toolbox = base.Toolbox()
        self.toolbox.register('attr_bool', random.randint, 0, 1)
        self.toolbox.register('individual', tools.initRepeat,
                              creator.Individual, self.toolbox.attr_bool, bqm_list[0].num_variables)
        self.toolbox.register('population', tools.initRepeat,
                              list, self.toolbox.individual)

        self.toolbox.register('evaluate', self._calc_ene, bqm_list=bqm_list)
        self.toolbox.register('mate', tools.cxTwoPoint)
        self.toolbox.register('mutate', tools.mutFlipBit, indpb=0.05)
        self.toolbox.register('select', tools.selTournament, tournsize=3)

    def sample(self, bqm_list, num_pops, cxpb, mutpb, num_gens):
        creator.create('FitnessMin', base.Fitness,
                       weights=tuple([-1.0] * len(bqm_list)))
        creator.create('Individual', list, fitness=creator.FitnessMin)
        self._regist_toolbox(bqm_list)

        pop = self.toolbox.population(n=num_pops)
        pop = self._apply_fit(pop)

        for g in range(num_gens):
            offspring = list(
                map(self.toolbox.clone, self.toolbox.select(pop, len(pop))))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                if random.random() < mutpb:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            invalid_ind = self._apply_fit(invalid_ind)
            pop[:] = offspring

        samples, frequencies = np.unique(pop, axis=0, return_counts=True)
        responses = [dimod.SampleSet.from_samples(dimod.as_samples(
            samples), dimod.BINARY, bqm.energies(samples), num_occurrences=frequencies) for bqm in bqm_list]

        return responses
