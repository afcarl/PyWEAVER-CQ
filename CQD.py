import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class CQDlayer(object):

    def __init__(self,
                 feedforward=.1,
                 feedback=-.1,
                 plan_competition=.0,
                 choice_competition=.0,
                 plan_offset=.0,
                 choice_offset=.0,
                 sustain=.9,
                 duration=1000,
                 input_duration=100,
                 output_offset=50,
                 activation='linear',
                 **input_dict):
        num_inputs = len(input_dict.keys())
        self.max_ticks = duration
        self.output_offset = output_offset
        self.input_duration = input_duration
        self.activation_function = activation
        self.input_idx = tuple(range(num_inputs))
        self.plan_idx = tuple(range(num_inputs, 2 * num_inputs))
        self.choice_idx = tuple(range(2 * num_inputs, 3 * num_inputs))
        width = len(self.input_idx) + len(self.plan_idx) + len(self.choice_idx)
        self.activations = np.zeros((self.max_ticks, width))
        # initialize weight matrix (and fill it up?)
        self.weights = np.zeros((width, width))
        plan_decay = sustain
        choice_decay = sustain
        # set input weight
        input_weights = []
        for i in range(num_inputs):
            input_weights.append(input_dict['input' + str(i)])
        input_weights = np.cumprod(input_weights)
        for i in range(len(self.input_idx)):
            self.weights[self.plan_idx[i], self.input_idx[i]] = input_weights[i]
        # set feedforward
        for i in range(len(self.plan_idx)):
            self.weights[self.choice_idx[i], self.plan_idx[i]] = feedforward
        # set feedback
        for i in range(len(self.choice_idx)):
            self.weights[self.plan_idx[i], self.choice_idx[i]] = feedback
        # set competition
        for i in self.plan_idx:
            for j in self.plan_idx:
                self.weights[i, j] = plan_competition
        for i in self.choice_idx:
            for j in self.choice_idx:
                self.weights[i, j] = choice_competition
        # set decay
        for i in self.plan_idx:
            self.weights[i, i] = plan_decay
        for i in self.choice_idx:
            self.weights[i, i] = choice_decay
        # start clock at 0
        self.ticks = 0
        # build input array
        if self.activation_function == 'sigmoid':
            self.inputs = np.zeros(self.activations.shape) - 100.0
            self.inputs[0:self.input_duration, 0:num_inputs] = 100.0
            self.inputs[:, self.plan_idx] = plan_offset
            self.inputs[:, self.choice_idx] = choice_offset
        else:
            self.inputs = np.zeros(self.activations.shape)
            self.inputs[0:self.input_duration, 0:num_inputs] = 1.0
            self.inputs[:, self.plan_idx] = plan_offset
            self.inputs[:, self.choice_idx] = choice_offset

    def update(self):
        # add activations and input together before multiplying by weights?
        # or apply weights to activations first and then add input?
        # self.activations[self.ticks + 1] = np.clip(np.sum((self.activations[self.ticks] + self.inputs[self.ticks])
        #                                                   * self.weights, axis=1), 0, 1)
        if self.activation_function == 'linear':
            self.activations[self.ticks + 1] = (np.sum((self.activations[self.ticks]) * self.weights, axis=1)
                                                + self.inputs[self.ticks])
        elif self.activation_function == 'clipped_linear':
            self.activations[self.ticks + 1] = np.clip(np.sum((self.activations[self.ticks]) * self.weights, axis=1)
                                                       + self.inputs[self.ticks], 0, 1)
        elif self.activation_function == 'ReLU':
            self.activations[self.ticks + 1] = ReLU(np.sum((self.activations[self.ticks]) * self.weights, axis=1)
                                                    + self.inputs[self.ticks])
        elif self.activation_function == 'sigmoid':
            self.activations[self.ticks + 1] = sigmoid(np.sum((self.activations[self.ticks]) * self.weights, axis=1)
                                                       + self.inputs[self.ticks])
        elif self.activation_function == 'tanh':
            self.activations[self.ticks + 1] = tanh(np.sum((self.activations[self.ticks]) * self.weights, axis=1)
                                                    + self.inputs[self.ticks])
        self.ticks += 1

    def evaluate(self):
        for tick in range(self.max_ticks - 1):
            self.update()

    def loss(self):
        method = 'continuous'
        # method = 'thresh_binary'

        if method == 'thresh_binary':
            self.active = (self.activations[:, self.choice_idx] > .8) * -1.0
            self.inactive = (self.activations[:, self.choice_idx] < .2) * 1.0
            num_choice = len(self.choice_idx)
            for idx in range(num_choice):
                target_start = int(self.output_offset + ((self.input_duration / num_choice) * idx))
                target_end = int(self.output_offset + ((self.input_duration / num_choice) * (idx + 1)))
                self.active[target_start:target_end, idx] *= -10.0
                self.inactive[target_start:target_end, idx] *= -0.1
            self.scores = self.active + self.inactive
            self.score = np.sum(self.scores)
        elif method == 'continuous':
            self.active = self.activations[:, self.choice_idx]
            num_choice = len(self.choice_idx)
            for idx in range(num_choice):
                target_start = int(self.output_offset + ((self.input_duration / num_choice) * idx))
                target_end = int(self.output_offset + ((self.input_duration / num_choice) * (idx + 1)))
                # self.active[target_start:target_end, idx] = np.sqrt(self.active[target_start:target_end, idx]) * -2.0
                self.active[target_start:target_end, idx] = self.active[target_start:target_end, idx] * -2.0
            self.scores = self.active * -1.0
            self.score = np.sum(self.scores)
        return self.score

    def plot_weights(self):
        plt.clf()
        fig, ax = plt.subplots()
        sns.heatmap(self.weights, cmap='RdBu', center=0.0)
        plt.savefig('weights.png')
        return fig, ax

    def plot(self, alpha=.5):
        plt.clf()
        fig, ax = plt.subplots()
        for idx in self.input_idx:
            plt.plot(self.activations[0:self.max_ticks, idx], label=idx, alpha=alpha)
        plt.legend()
        plt.savefig('input_activations.png')
        plt.clf()
        fig, ax = plt.subplots()
        for idx in self.plan_idx:
            plt.plot(self.activations[0:self.max_ticks, idx], label=idx, alpha=alpha)
        plt.legend()
        plt.savefig('plan_activations.png')
        plt.clf()
        fig, ax = plt.subplots()
        for idx in self.choice_idx:
            plt.plot(self.activations[0:self.max_ticks, idx], label=idx, alpha=alpha)
        plt.legend()
        plt.savefig('choice_activations.png')
        return fig, ax


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x):
    return (2.0 / (1.0 + np.exp(-2.0 * x))) - 1.0


def ReLU(x):
    return np.clip(x, 0, None)
