import json
import data_preproc as dp
import nn
import pandas as pd
import layers


class GridSearch:

    def __init__(self, params_file):
        with open(params_file) as json_file:
            data = json.load(json_file)
            for config in data:
                self.init_and_train_network(config)
        return

    def load_data(self, data):
        data_loader = dp.DataLoader(data=data)
        return data_loader

    def init_and_train_network(self, params_net):
        data_loader = self.load_data(params_net['data'])
        use_bias = bool(params_net['use_bias'])
        neural_network = nn.Network(
            bias=use_bias,
            shape_in=pd.DataFrame(data_loader.train_x).shape).init_network(
            save_plot_values=True
        )
        for layer in params_net['hidden_layers']:
            if layer['type'] == 'dropout':
                neural_network.add_layer(
                    layer=layers.DropoutLayer(
                        dropout_prob=layer['prob'],
                        nr_neurons=layer['nr_neurons']
                    )
                )
            else:
                neural_network.add_layer(
                    size=layer['nr_neurons'],
                    activation=layer['activation'],
                    init_type=layer['init'])

        neural_network.add_output(
            data_loader.train_y.shape,
            params_net['out_activ'],
            params_net['loss'],
            init_type=params_net['out_init']
        )
        batch_size = params_net['batch_size']
        if params_net['online']:
            batch_size = len(data_loader.train_x)
        neural_network.train_network(data_loader.train_x,
                                     data_loader.train_y,
                                     data_loader.test_x, data_loader.test_y,
                                     online=params_net['online'],
                                     learning_rate=params_net['learning_rate'],
                                     nr_epochs=params_net['epochs'],
                                     batch_size=batch_size)
        return
