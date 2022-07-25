from ..data import GaussianIID
from ..model import LinearNet

def print_blank():
    L = 4
    d = 10
    m = 100
    q = 5
    alpha = 0.0
    beta = 100.0
    epsilon = 0.1






    net = LinearNet(num_layers= -1, 
            input_size=d, 
            hidden_size=m, 
            output_size=q, )

    weights = net.weights()

    data = GaussianIID(dimension = d, num_samples = 17, k = 3, epsilon = 0)
    breakpoint()
    print("here")
