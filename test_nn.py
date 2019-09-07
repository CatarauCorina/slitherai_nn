import pandas as pd
import sklearn.datasets as ds
from slitherai_nn.src import nn as nn


def main():
    x_iris, y_iris = ds.load_iris(return_X_y=True)
    neural_network = nn.Network(x_iris, y_iris).init_network().add_layer(3)
    output = neural_network.run()
    print(output)


if __name__ == '__main__':
    main()

