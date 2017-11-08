from CQD import CQDlayer
from optunity.parallel import create_pmap
from optunity.solvers import ParticleSwarm
from time import time


def optimize_CQD(search_params):
    optunity_params = dict()
    fixed_params = dict()
    for key, item in search_params.items():
        if type(item) is list:
            optunity_params[key] = item
        else:
            fixed_params[key] = item

    def optimize_wrapper(**test_params):
        test_params.update(fixed_params)
        cqd = CQDlayer(**test_params)
        cqd.evaluate()
        score = cqd.loss()
        if score > 0:
            print(int(score))
        return score

    solver = ParticleSwarm(num_particles=8000,
                           num_generations=500,
                           **optunity_params)
    params, _ = solver.maximize(optimize_wrapper, pmap=create_pmap(number_of_processes=4))
    params.update(fixed_params)
    return params


# search params
search_params = {
    'feedforward': [-1, 0],  # feedforward plan->choice, fraction of 1
    'feedback': [0, 1],  # feedback choice->plan, fraction of -1
    'plan_competition': 0,  # competitive inhibition in choice layer, fraction of -1
    'choice_competition': [-1, 0],  # competitive inhibition in choice layer, fraction of -1
    'plan_offset': [0, 1],  # bias for plan nodes
    'choice_offset': [0, 1],  # bias for choice nodes
    'sustain': [0, 1],  # 1 - decay in all nodes, fraction of 1
    'duration': 500,  # time interval for model evaluation
    'input_duration': 300,  # duration of input signal at word node, signal amplitude is 1
    'output_offset': 25,  # delay before output is supposed to start
    'activation': 'clipped_linear',  # activation function, options are linear, sigmoid, and tanh?
    'input0': [-1, 0],  # input gradient 1, fraction of 1
    'input1': [0, 1],  # input gradient 2, fraction of input gradient 1
    'input2': [0, 1]  # input gradient 3, fraction of input gradient 2
}

t0 = time()
params = optimize_CQD(search_params)
cqd = CQDlayer(**params)
cqd.plot_weights()
cqd.evaluate()
cqd.plot()
score = cqd.loss()
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
