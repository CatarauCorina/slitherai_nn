import json
from datetime import datetime as dt


class Plotter:
    activations = {}
    accuracy_epochs = {}
    out_prob = {}
    loss_epochs = {}
    count = 0

    def add_activation(self, value, layer):
        if layer in self.activations:
            self.activations[layer].append(value.tolist())
        else:
            self.activations[layer] = [value.tolist()]
        self.count += 1
        if self.count == 100:
            self.dump_values('activations')
        return

    def add_out_prob(self, iter, value):
        if iter in self.out_prob:
            self.out_prob[iter].append(value.tolist())
        else:
            self.out_prob[iter] = [value.tolist()]
        self.count += 1
        if self.count == 100:
            self.dump_values('out_prob')
        return

    def dump_values(self, choice):
        values_to_dump = getattr(self, choice)
        now = str(dt.now().date())
        with open(f'{choice}_{now}.json', 'w') as fp:
            json.dump(values_to_dump, fp)
        return
