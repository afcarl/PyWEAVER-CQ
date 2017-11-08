import numpy as np
from CQ import CQlayer


# parameters
params = {
    'feedforward': 0.17105904387620408,  # feedforward plan->choice, fraction of 1
    'feedback': -0.44397795828846265,  # feedback choice->plan, fraction of -1
    'plan_competition': -0.3949908131397458,  # competitive inhibition in choice layer, fraction of -1
    'choice_competition': -0.24803265677471717,  # competitive inhibition in choice layer, fraction of -1
    'sustain': 0.993215486939317,  # 1 - decay in all nodes, fraction of 1
    'duration': 500,  # time interval for model evaluation
    'input_duration': 300,  # duration of input signal at word node, signal amplitude is 1
    'output_offset': 25,  # delay before output is supposed to start
    'activation': 'clipped_linear',  # activation function, options are linear, sigmoid, and tanh?
    'input0': 0.8337507932889009,  # input gradient 1, fraction of 1
    'input1': 0.9476350912913027,  # input gradient 2, fraction of input gradient 1
    'input2': 0.8591204400358125  # input gradient 3, fraction of input gradient 2
}
cq = CQlayer(**params)
cq.plot_weights()
cq.evaluate()
cq.plot()
