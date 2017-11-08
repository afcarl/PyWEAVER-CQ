from LIF import LIFlayer
from optunity.parallel import create_pmap
from optunity.solvers import ParticleSwarm
from time import time


def optimize_LIF(search_params):
    optunity_params = dict()
    fixed_params = dict()
    for key, item in search_params.items():
        if type(item) is list:
            optunity_params[key] = item
        else:
            fixed_params[key] = item

    def optimize_wrapper(**test_params):
        test_params.update(fixed_params)
        lif = LIFlayer(**test_params)
        lif.evaluate()
        score = lif.loss()
        if score != 0:
            print(int(score))
        return score

    solver = ParticleSwarm(num_particles=800,
                           num_generations=50,
                           **optunity_params)
    params, _ = solver.maximize(optimize_wrapper, pmap=create_pmap(number_of_processes=4))
    params.update(fixed_params)
    return params

# search params
search_params = {
    'I': [0, 5],
    'C_exp': [1, 6],
    'R': [8, 100],
    'input0': [0, 1],  # input gradient 1, fraction of 1
    'input1': [0, 1],  # input gradient 2, fraction of input gradient 1
    'input2': [0, 1]  # input gradient 3, fraction of input gradient 2
}

t0 = time()
params = optimize_LIF(search_params)
lif = LIFlayer(**params)
lif.evaluate()
lif.plot()
score = lif.loss()
t1 = time()
print('optimal parameters:')
for key, item in params.items():
    print(key + ': ' + str(item))
dur = t1 - t0
if dur < 60.0:
    print('running time: ' + str(int(dur)) + ' sec')
else:
    print('running time: ' + str(int(dur / 60.0)) + ' min')
print('score: ' + str(int(score)))
