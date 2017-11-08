import numpy as np
from CQD import CQDlayer


# parameters
params = {
    'feedforward': -.08,  # feedforward plan->choice, fraction of 1
    'feedback': .08,  # feedback choice->plan, fraction of -1
    'plan_competition': 0.0,  # competitive inhibition in choice layer, fraction of -1
    'choice_competition': -.05,  # competitive inhibition in choice layer, fraction of -1
    'sustain': 1.0,  # 1 - decay in all nodes, fraction of 1
    'duration': 500,  # time interval for model evaluation
    'choice_offset': .04,
    'plan_offset': .06,
    'input_duration': 300,  # duration of input signal at word node, signal amplitude is 1
    'output_offset': 25,  # delay before output is supposed to start
    'activation': 'clipped_linear',  # activation function, options are linear, sigmoid, and tanh?
    'input0': -0.11,  # input gradient 1, fraction of 1
    'input1': -0.10,  # input gradient 2, fraction of input gradient 1
    'input2': -0.09  # input gradient 3, fraction of input gradient 2
}
cqd = CQDlayer(**params)
cqd.plot_weights()
cqd.evaluate()
cqd.plot()
