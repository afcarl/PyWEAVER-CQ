import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class LIFlayer(object):

    def __init__(self,
                 ms=500,
                 dt=.1,
                 I=1.5,
                 R=1.0,
                 C_exp=2.0,
                 **input_dict):
        self.input_dict = input_dict
        self.num_inputs = len(input_dict.keys())
        self.max_ticks = int(ms / dt)
        self.dt = dt
        self.I = I
        self.R = R
        self.C = 1.0 * np.power(10.0, C_exp)
        self.output_offset = int(25 / dt)
        self.input_duration = int(100 / dt)
        self.inputs = np.zeros((self.max_ticks, self.num_inputs))
        self.inputs[:self.num_inputs * self.input_duration] = self.I
        input_weights = []
        for i in range(self.num_inputs):
            input_weights.append(input_dict['input' + str(i)])
        input_weights = np.cumprod(input_weights)
        for i in range(self.num_inputs):
            self.inputs[:, i] *= self.inputs[:, i] * input_weights[i]
        self.activations = np.zeros((self.max_ticks, self.num_inputs))
        self.spikes = np.zeros((self.max_ticks, self.num_inputs))
        self.ticks = 0

    def update(self):
        # Vm[i] = Vm[i - 1] + (-Vm[i - 1] + I * Rm) / tau_m * dt
        # Vm[i] = Vm[i - 1] + (-Vm[i - 1] + input * R) / tau_m * dt
        self.activations[self.ticks + 1] = (self.activations[self.ticks] + (-self.activations[self.ticks]
                                                                                  + (self.inputs[self.ticks] * self.R))
                                                  / ((self.R * self.C) * self.dt))
        self.spikes[self.ticks + 1] = self.activations[self.ticks + 1] >= 1.0
        self.activations[self.ticks + 1] *= ((self.spikes[self.ticks + 1] * -1.0) + 1.0)
        self.ticks += 1

    def evaluate(self):
        for tick in range(self.max_ticks - 1):
            self.update()

    def loss(self):
        self.active = self.spikes
        for idx in range(self.num_inputs):
            target_start = int(self.output_offset + ((self.input_duration / self.num_inputs) * idx))
            target_end = int(self.output_offset + ((self.input_duration / self.num_inputs) * (idx + 1)))
            self.active[target_start:target_end, idx] = self.active[target_start:target_end, idx] * -5.0
        self.scores = self.active * -1.0
        self.score = np.sum(self.scores)
        return self.score

    def plot(self, alpha=.5):
        plt.clf()
        fig, ax = plt.subplots()
        for idx in range(self.num_inputs):
            plt.plot(self.activations[0:self.max_ticks, idx], label=idx, alpha=alpha)
        plt.legend()
        plt.savefig('LIF_activations.png')
        plt.clf()
        fig, ax = plt.subplots()
        for idx in range(self.num_inputs):
            plt.plot(self.spikes[0:self.max_ticks, idx], label=idx, alpha=alpha)
        plt.legend()
        plt.savefig('LIF_spikes.png')
        return fig, ax
