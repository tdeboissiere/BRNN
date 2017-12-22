import math
import torch
import warnings

from helpers import compute_KL
from torch.nn import Module
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import PackedSequence


class BayesEmbedding(Module):

    def __init__(self, num_embeddings, embedding_dim,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 sparse=False, log_sigma1=-1.0, log_sigma2=-7.0, prior_pi=0.25, var_mode="BBB"):
        super(BayesEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.var_mode = var_mode
        if self.var_mode == "BBB":
            self.weight_mu = Parameter(torch.Tensor(num_embeddings, embedding_dim))
            self.weight_rho = Parameter(torch.Tensor(num_embeddings, embedding_dim))
        else:
            self.weight = Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.sparse = sparse

        self.log_sigma1 = log_sigma1
        self.log_sigma2 = log_sigma2
        self.sigma1 = math.exp(log_sigma1)
        self.sigma2 = math.exp(log_sigma2)
        self.prior_pi = prior_pi
        self.sigma_mix = math.sqrt(prior_pi * self.sigma1 ** 2 + (1.0 - prior_pi) * self.sigma2 ** 2)
        self.rho_max_init = math.log(math.exp(self.sigma_mix / 1.0) - 1.0)
        self.rho_min_init = math.log(math.exp(self.sigma_mix / 2.0) - 1.0)

        # Default value
        self.padding_idx = -1
        self.kl = None
        self.reset_parameters()

    def reset_parameters(self):
        initrange = 0.05
        if self.var_mode == "BBB":
            self.weight_mu.data.uniform_(-initrange, initrange)
            self.weight_rho.data.uniform_(self.rho_min_init, self.rho_max_init)
        else:
            self.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):

        if self.var_mode == "BBB":

            # Sample weight
            mean = self.weight_mu
            sigma = torch.nn.functional.softplus(self.weight_rho) + 1E-5

            # print("Emb", sigma.data.cpu().numpy().min(), sigma.data.cpu().numpy().max())

            eps = Variable(torch.randn(mean.size()).type_as(mean.data))
            weights = mean + eps * sigma

            # Compute KL divergence
            self.kl = compute_KL(weights.view(-1), mean.view(-1), sigma.view(-1),
                                 self.sigma1, self.sigma2, self.prior_pi)

        else:
            self.kl = 0
            weights = self.weight

        return self._backend.Embedding.apply(
            input, weights,
            self.padding_idx, self.max_norm, self.norm_type,
            self.scale_grad_by_freq, self.sparse
        )

    def __repr__(self):
        s = '{name}({num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class BayesLinear(Module):

    def __init__(self, in_features, out_features, bias=True, log_sigma1=-1.0, log_sigma2=-7.0, prior_pi=0.25, var_mode="BBB"):
        super(BayesLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.log_sigma1 = log_sigma1
        self.log_sigma2 = log_sigma2
        self.sigma1 = math.exp(log_sigma1)
        self.sigma2 = math.exp(log_sigma2)
        self.prior_pi = prior_pi
        self.sigma_mix = math.sqrt(prior_pi * self.sigma1 ** 2 + (1.0 - prior_pi) * self.sigma2 ** 2)
        self.rho_max_init = math.log(math.exp(self.sigma_mix / 1.0) - 1.0)
        self.rho_min_init = math.log(math.exp(self.sigma_mix / 2.0) - 1.0)

        self.var_mode = var_mode

        if self.var_mode == "BBB":
            self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
            self.weight_rho = Parameter(torch.Tensor(out_features, in_features))
        else:
            self.weight = Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.kl = None

        self.reset_parameters()

    def reset_parameters(self):

        initrange = 0.05

        if self.var_mode == "BBB":
            self.weight_mu.data.uniform_(-initrange, initrange)
            self.weight_rho.data.uniform_(self.rho_min_init, self.rho_max_init)
            if self.bias is not None:
                self.bias.data.fill_(0)

        else:
            if self.bias is not None:
                self.bias.data.fill_(0)
            self.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):

        if self.var_mode == "BBB":
            # Sample weight
            mean = self.weight_mu
            sigma = torch.nn.functional.softplus(self.weight_rho) + 1E-5

            # print("Linear", sigma.data.cpu().numpy().min(), sigma.data.cpu().numpy().max())

            # Sample weights from normal distribution
            eps = Variable(torch.randn(mean.size()).type_as(mean.data))
            weights = mean + eps * sigma

            # Compute KL divergence
            self.kl = compute_KL(weights.view(-1), mean.view(-1), sigma.view(-1),
                                 self.sigma1, self.sigma2, self.prior_pi)

        else:
            self.kl = 0
            weights = self.weight

        return F.linear(input, weights, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class BayesRNNBase(Module):

    def __init__(self, mode, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False, log_sigma1=-1.0, log_sigma2=-7.0, prior_pi=0.25, var_mode="BBB"):
        super(BayesRNNBase, self).__init__()
        self.mode = mode
        self.var_mode = var_mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.dropout_state = {}
        self.bidirectional = bidirectional
        self.log_sigma1 = log_sigma1
        self.log_sigma2 = log_sigma2
        self.sigma1 = math.exp(log_sigma1)
        self.sigma2 = math.exp(log_sigma2)
        self.prior_pi = prior_pi
        self.sigma_mix = math.sqrt(prior_pi * self.sigma1 ** 2 + (1.0 - prior_pi) * self.sigma2 ** 2)
        self.rho_max_init = math.log(math.exp(self.sigma_mix / 2.0) - 1.0)
        self.rho_min_init = math.log(math.exp(self.sigma_mix / 4.0) - 1.0)
        num_directions = 2 if bidirectional else 1

        if mode == 'LSTM':
            gate_size = 4 * hidden_size
        elif mode == 'GRU':
            gate_size = 3 * hidden_size
        else:
            gate_size = hidden_size

        self._all_weights = []
        self._param_buf_size = 0
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions

                b_ih = Parameter(torch.Tensor(gate_size))
                b_hh = Parameter(torch.Tensor(gate_size))

                if self.var_mode == "BBB":

                    w_ih_mu = Parameter(torch.Tensor(gate_size, layer_input_size))
                    w_ih_rho = Parameter(torch.Tensor(gate_size, layer_input_size))
                    w_hh_mu = Parameter(torch.Tensor(gate_size, hidden_size))
                    w_hh_rho = Parameter(torch.Tensor(gate_size, hidden_size))

                    layer_params = (w_ih_mu, w_ih_rho, w_hh_mu, w_hh_rho, b_ih, b_hh)

                    suffix = '_reverse' if direction == 1 else ''
                    param_names = ['weight_ih_mu_l{}{}', 'weight_ih_rho_l{}{}',
                                   'weight_hh_mu_l{}{}', 'weight_hh_rho_l{}{}']

                    if bias:
                        param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                    param_names = [x.format(layer, suffix) for x in param_names]

                    for name, param in zip(param_names, layer_params):
                        setattr(self, name, param)
                    self._all_weights.append(param_names)

                    self._param_buf_size += sum(p.numel() for p in layer_params)

                else:

                    w_ih = Parameter(torch.Tensor(gate_size, layer_input_size))
                    w_hh = Parameter(torch.Tensor(gate_size, hidden_size))

                    layer_params = (w_ih, w_hh, b_ih, b_hh)

                    suffix = '_reverse' if direction == 1 else ''
                    param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                    if bias:
                        param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                    param_names = [x.format(layer, suffix) for x in param_names]

                    for name, param in zip(param_names, layer_params):
                        setattr(self, name, param)
                    self._all_weights.append(param_names)

                    self._param_buf_size += sum(p.numel() for p in layer_params)

        if self.var_mode == "BBB":

            # Removed flattened parameters for now
            # So need to manually set _data_ptrs
            # This triggers the warning :
            # bayesian_rnn_pt.py:32: UserWarning: RNN module weights are not part of single contiguous chunk of memory.
            # This means they need to be compacted at every call, possibly greately increasing memory usage.
            # To compact weights again call flatten_parameters().
            # The reason flatten parameters was removed is that with BayesRNN, we have more parameters than usual (x2)
            # which led to a size mismatch in flatten parameters
            self._data_ptrs = []

        else:
            self.flatten_parameters()

        self.reset_parameters()
        self.kl = None

    def reset_parameters(self):

        initrange = 0.05

        if self.var_mode == "BBB":
            for name, param in self.named_parameters():
                if "rho" in name:
                    param.data.uniform_(self.rho_min_init, self.rho_max_init)
                elif "bias" in name:
                    param.data.fill_(0.)
                else:
                    param.data.uniform_(-initrange, initrange)

        else:
            stdv = 1.0 / math.sqrt(self.hidden_size)
            for weight in self.parameters():
                weight.data.uniform_(-stdv, stdv)

    def flatten_parameters(self):
        """Resets parameter data pointer so that they can use faster code paths.

        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
        any_param = next(self.parameters()).data
        if not any_param.is_cuda or not torch.backends.cudnn.is_acceptable(any_param):
            self._data_ptrs = []
            return

        # This is quite ugly, but it allows us to reuse the cuDNN code without larger
        # modifications. It's really a low-level API that doesn't belong in here, but
        # let's make this exception.
        from torch.backends.cudnn import rnn
        from torch.backends import cudnn
        from torch.nn._functions.rnn import CudnnRNN
        handle = cudnn.get_handle()
        with warnings.catch_warnings(record=True):
            fn = CudnnRNN(
                self.mode,
                self.input_size,
                self.hidden_size,
                num_layers=self.num_layers,
                batch_first=self.batch_first,
                dropout=self.dropout,
                train=self.training,
                bidirectional=self.bidirectional,
                dropout_state=self.dropout_state,
            )

        # Initialize descriptors
        fn.datatype = cudnn._typemap[any_param.type()]
        fn.x_descs = cudnn.descriptor(any_param.new(1, self.input_size), 1)
        fn.rnn_desc = rnn.init_rnn_descriptor(fn, handle)

        # Allocate buffer to hold the weights
        num_weights = rnn.get_num_weights(handle, fn.rnn_desc, fn.x_descs[0], fn.datatype)
        fn.weight_buf = any_param.new(num_weights).zero_()
        fn.w_desc = rnn.init_weight_descriptor(fn, fn.weight_buf)

        # Slice off views into weight_buf
        params = rnn.get_parameters(fn, handle, fn.weight_buf)
        all_weights = [[p.data for p in l] for l in self.all_weights]

        # Copy weights and update their storage
        rnn._copyParams(all_weights, params)
        for orig_layer_param, new_layer_param in zip(all_weights, params):
            for orig_param, new_param in zip(orig_layer_param, new_layer_param):
                orig_param.set_(new_param.view_as(orig_param))

        self._data_ptrs = list(p.data.data_ptr() for p in self.parameters())

    def forward(self, input, hx=None):
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes = input
            max_batch_size = batch_sizes[0]
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.autograd.Variable(input.data.new(self.num_layers *
                                                        num_directions,
                                                        max_batch_size,
                                                        self.hidden_size).zero_())
            if self.mode == 'LSTM':
                hx = (hx, hx)

        has_flat_weights = list(p.data.data_ptr() for p in self.parameters()) == self._data_ptrs
        if has_flat_weights:
            first_data = next(self.parameters()).data
            assert first_data.storage().size() == self._param_buf_size
            flat_weight = first_data.new().set_(first_data.storage(), 0, torch.Size([self._param_buf_size]))
        else:
            flat_weight = None
        func = self._backend.RNN(
            self.mode,
            self.input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            dropout=self.dropout,
            train=self.training,
            bidirectional=self.bidirectional,
            batch_sizes=batch_sizes,
            dropout_state=self.dropout_state,
            flat_weight=flat_weight
        )

        # Format weights for BBB
        if self.var_mode == "BBB":
            all_weights = []
            num_layers = self.num_layers
            num_directions = 2 if self.bidirectional else 1
            # Loop over layers
            for layer_idx in range(num_layers):
                # Loop over directions
                for direction in range(num_directions):

                    suffix = "_reverse" if direction == 1 else ""

                    mean_ih = getattr(self, f"weight_ih_mu_l{layer_idx}{suffix}")
                    rho_ih = getattr(self, f"weight_ih_rho_l{layer_idx}{suffix}")
                    sigma_ih = torch.nn.functional.softplus(rho_ih) + 1E-5

                    mean_hh = getattr(self, f"weight_hh_mu_l{layer_idx}{suffix}")
                    rho_hh = getattr(self, f"weight_hh_rho_l{layer_idx}{suffix}")
                    sigma_hh = torch.nn.functional.softplus(rho_hh) + 1E-5

                    # print(f"ih_l{layer_idx}{suffix}", sigma_ih.data.cpu().numpy().min(), sigma_ih.data.cpu().numpy().max())
                    # print(f"hh_l{layer_idx}{suffix}", sigma_hh.data.cpu().numpy().min(), sigma_hh.data.cpu().numpy().max())

                    # Sample weights from normal distribution
                    eps_ih = Variable(torch.randn(mean_ih.size()).type_as(mean_ih.data))
                    weight_ih = mean_ih + eps_ih * sigma_ih

                    eps_hh = Variable(torch.randn(mean_hh.size()).type_as(mean_hh.data))
                    weight_hh = mean_hh + eps_hh * sigma_hh

                    # Compute KL divergence
                    self.kl = compute_KL(weight_ih.view(-1), mean_ih.view(-1), sigma_ih.view(-1),
                                         self.sigma1, self.sigma2, self.prior_pi)
                    self.kl += compute_KL(weight_hh.view(-1), mean_hh.view(-1), sigma_hh.view(-1),
                                          self.sigma1, self.sigma2, self.prior_pi)

                    weights = [weight_ih, weight_hh]

                    # Get biases
                    if self.bias:
                        bias_ih = getattr(self, f"bias_ih_l{layer_idx}{suffix}")
                        bias_hh = getattr(self, f"bias_hh_l{layer_idx}{suffix}")

                        weights += [bias_ih, bias_hh]

                    all_weights.append(weights)
        else:
            all_weights = self.all_weights
            self.kl = 0

        output, hidden = func(input, all_weights, hx)
        if is_packed:
            output = PackedSequence(output, batch_sizes)
        return output, hidden

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def __setstate__(self, d):
        super(BayesRNNBase, self).__setstate__(d)
        self.__dict__.setdefault('_data_ptrs', [])
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        num_layers = self.num_layers
        num_directions = 2 if self.bidirectional else 1
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix = '_reverse' if direction == 1 else ''
                weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}', 'bias_ih_l{}{}', 'bias_hh_l{}{}']
                weights = [x.format(layer, suffix) for x in weights]
                if self.bias:
                    self._all_weights += [weights]
                else:
                    self._all_weights += [weights[:2]]

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]


class BayesRNN(BayesRNNBase):

    def __init__(self, *args, **kwargs):
        if 'nonlinearity' in kwargs:
            if kwargs['nonlinearity'] == 'tanh':
                mode = 'RNN_TANH'
            elif kwargs['nonlinearity'] == 'relu':
                mode = 'RNN_RELU'
            else:
                raise ValueError("Unknown nonlinearity '{}'".format(
                    kwargs['nonlinearity']))
            del kwargs['nonlinearity']
        else:
            mode = 'RNN_TANH'

        super(BayesRNN, self).__init__(mode, *args, **kwargs)


class BayesLSTM(BayesRNNBase):

    def __init__(self, *args, **kwargs):
        super(BayesLSTM, self).__init__('LSTM', *args, **kwargs)


if __name__ == '__main__':

    lstm = BayesLSTM(10,20, 2, bidirectional=True)
    inp = torch.autograd.Variable(torch.ones((5,32,10)))
    print(lstm._all_weights)
    lstm(inp)
    # lstm.all_weights()
