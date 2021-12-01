import graph
import torch

class FCLayer(torch.nn.Module):
    r"""
    A simple fully connected and customizable layer. This layer is centered around a torch.nn.Linear module.
    The order in which transformations are applied is:
    #. Dense Layer
    #. Activation
    #. Dropout (if applicable)
    #. Batch Normalization (if applicable)
    Arguments
    ----------
        in_size: int
            Input dimension of the layer (the torch.nn.Linear)
        out_size: int
            Output dimension of the layer.
        dropout: float, optional
            The ratio of units to dropout. No dropout by default.
            (Default value = 0.)
        activation: str or callable, optional
            Activation function to use.
            (Default value = relu)
        b_norm: bool, optional
            Whether to use batch normalization
            (Default value = False)
        bias: bool, optional
            Whether to enable bias in for the linear layer.
            (Default value = True)
        init_fn: callable, optional
            Initialization function to use for the weight of the layer. Default is
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` with :math:`k=\frac{1}{ \text{in_size}}`
            (Default value = None)
    Attributes
    ----------
        dropout: int
            The ratio of units to dropout.
        b_norm: int
            Whether to use batch normalization
        linear: torch.nn.Linear
            The linear layer
        activation: the torch.nn.Module
            The activation layer
        init_fn: function
            Initialization function used for the weight of the layer
        in_size: int
            Input dimension of the linear layer
        out_size: int
            Output dimension of the linear layer
    """

    def __init__(self, in_size, out_size, activation, dropout=0., b_norm=False, bias=True):
        super(FCLayer, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.bias = bias
        self.linear = torch.nn.Linear(in_size, out_size, bias=bias)
        self.dropout = None
        self.b_norm = None
        if dropout:
            self.dropout = torch.nn.Dropout(p=dropout)
        if b_norm:
            self.b_norm = torch.nn.BatchNorm1d(out_size)
        self.activation = activation

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)
        if self.b_norm is not None:
            h = self.b_norm(h)
        return h
    
class style_Dense(torch.nn.Module):
    def __init__(self, dims, activation_type, drop=0., batch_norm=False):
        super(style_Dense, self).__init__()
        self.depth = len(dims) - 1
        self.flat = torch.nn.Flatten()
        self.dense_layers = torch.nn.ModuleList([])
        for i in range(self.depth - 1):
            self.dense_layers.append(FCLayer(dims[i], dims[i+1], activation_type(), drop, batch_norm))
        self.dense_layers.append(FCLayer(dims[-2], dims[-1], None, drop, False))

    def forward(self, x):
        x = self.flat(x)
        for layer in self.dense_layers:
            x = layer(x)
        return x

class Classifier_Dense(torch.nn.Module):
    def __init__(self, dims, activation_type):
        super(Classifier_Dense, self).__init__()
        self.depth = len(dims) - 1
        self.flat = torch.nn.Flatten()
        self.dense_layers = torch.nn.ModuleList([])
        self.activations = torch.nn.ModuleList([])
        for i in range(self.depth - 1):
            self.dense_layers.append(torch.nn.Linear(dims[i], dims[i+1]))
            self.activations.append(activation_type())
        self.dense_layers.append(torch.nn.Linear(dims[-2], dims[-1]))

        self.drop = torch.nn.Dropout(0.1)
    
    
    def forward(self, x):
        x = self.flat(x)
        for i in range(self.depth - 3):
            x = self.dense_layers[i](x)
            x = self.activations[i](x)
            #x = self.drop(x)
        for i in range(self.depth - 3, self.depth - 1):
            x = self.dense_layers[i](x)
            x = self.activations[i](x)
        x = self.dense_layers[-1](x)
        return x

class Classifier_RNN(torch.nn.Module):
    def __init__(self, dims, activation_type, gnn_output_dim):
        super(Classifier_RNN, self).__init__()
        self.depth = len(dims) - 1
        self.dims = dims
        self.gnn_output_dim = gnn_output_dim
        self.d0 = torch.nn.Linear(2*gnn_output_dim, 2*gnn_output_dim)
        self.acti0 = torch.nn.ReLU()
        self.lstm = torch.nn.GRU(
            input_size=gnn_output_dim*2,
            hidden_size=dims[1],
            num_layers=1,
            batch_first=True
        )
        self.drop = torch.nn.Dropout(0.2)
        
        self.dense_layers = torch.nn.ModuleList([])
        self.activations = torch.nn.ModuleList([])
        for i in range(1, self.depth - 1):
            self.dense_layers.append(torch.nn.Linear(dims[i], dims[i+1]))
            self.activations.append(activation_type())
        self.dense_layers.append(torch.nn.Linear(dims[-2], dims[-1]))
    
    def forward(self, x):
        x  = torch.stack(torch.split(x, self.gnn_output_dim, -1), 1)
        x = torch.flatten(x, -2)
        x = self.acti0(x + self.d0(x))
        x, _ = self.lstm(x)
        x = x[:, -1, :]

        for i in range(self.depth - 4):
            x = self.dense_layers[i](x)
            x = self.activations[i](x)
            x = self.drop(x)
        for i in range(self.depth - 4, self.depth-2):
            x = self.dense_layers[i](x)
            x = self.activations[i](x)
        x = self.dense_layers[-1](x)

        return x
