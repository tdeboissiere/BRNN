import math
import torch

from helpers import compute_KL
from torch.nn import Module
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class Prior(object):
    def __init__(self, pi, log_sigma1, log_sigma2):
        self.pi_mixture = pi
        self.log_sigma1 = log_sigma1
        self.log_sigma2 = log_sigma2
        self.sigma1 = math.exp(log_sigma1)
        self.sigma2 = math.exp(log_sigma2)

        self.sigma_mix = math.sqrt(pi * math.pow(self.sigma1, 2) + (1.0 - pi) * math.pow(self.sigma2, 2))

    def lstm_init(self):
        """Returns parameters to use when initializing \theta in the LSTM"""
        rho_max_init = math.log(math.exp(self.sigma_mix / 2.0) - 1.0)
        rho_min_init = math.log(math.exp(self.sigma_mix / 4.0) - 1.0)
        return rho_min_init, rho_max_init

    def normal_init(self):
        """Returns parameters to use when initializing \theta in embedding/projection layer"""
        rho_max_init = math.log(math.exp(self.sigma_mix / 1.0) - 1.0)
        rho_min_init = math.log(math.exp(self.sigma_mix / 2.0) - 1.0)
        return rho_min_init, rho_max_init


def get_bbb_variable(shape, prior, init_scale, rho_min_init, rho_max_init):

    if rho_min_init is None or rho_max_init is None:
        rho_max_init = math.log(math.exp(prior.sigma_mix / 2.0) - 1.0)
        rho_min_init = math.log(math.exp(prior.sigma_mix / 4.0) - 1.0)

    mu = Parameter(torch.Tensor(*shape))
    rho = Parameter(torch.Tensor(*shape))

    # Initialize
    mu.data.uniform_(-init_scale, init_scale)
    rho.data.uniform_(rho_min_init, rho_max_init)

    return mu, rho


class BayesLSTM(Module):

    def __init__(self, input_size, hidden_size, num_layers, prior, init_scale):

        super(BayesLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.prior = prior
        self.init_scale = init_scale
        self.num_layers = num_layers
        self._forget_bias = 1.0

        rho_min_init, rho_max_init = self.prior.lstm_init()

        for layer_idx in range(num_layers):

            # if layer_idx == 0:

            #     mu, rho = get_bbb_variable((self.input_size + self.hidden_size, 4 * self.hidden_size),
            #                                self.prior,
            #                                self.init_scale,
            #                                rho_min_init,
            #                                rho_max_init)
            # else:
            #     mu, rho = get_bbb_variable((self.hidden_size + self.hidden_size, 4 * self.hidden_size),
            #                                self.prior,
            #                                self.init_scale,
            #                                rho_min_init,
            #                                rho_max_init)

            mu, rho = get_bbb_variable((self.hidden_size + self.hidden_size, 4 * self.hidden_size),
                                       self.prior,
                                       self.init_scale,
                                       rho_min_init,
                                       rho_max_init)

            bias = Parameter(torch.Tensor(4 * self.hidden_size))
            bias.data.fill_(0.)

            setattr(self, f"mu_l{layer_idx}", mu)
            setattr(self, f"rho_l{layer_idx}", rho)
            setattr(self, f"bias_l{layer_idx}", bias)

        self.kl = None

    def forward_layer(self, x, hidden, layer_idx):

        # Get BBB params
        mean = getattr(self, f"mu_l{layer_idx}")
        sigma = F.softplus(getattr(self, f"rho_l{layer_idx}")) + 1E-5
        # Sample weights
        eps = Variable(torch.randn(mean.size()).type_as(mean.data))
        weights = mean + eps * sigma

        # Roll out hidden
        h, c = hidden[0][layer_idx], hidden[1][layer_idx]

        # Store each hidden state in output
        output = []
        # Loop over time steps and obtain predictions
        for i in range(x.size(0)):

            concat = torch.cat([x[i], h], -1)
            concat = torch.matmul(concat, weights) + getattr(self, f"bias_l{layer_idx}")

            i, j, f, o = torch.split(concat, concat.size(1) // 4, dim=1)

            new_c = c * F.sigmoid(f + self._forget_bias) + F.sigmoid(i) * F.tanh(j)
            new_h = F.tanh(new_c) * F.sigmoid(o)

            h, c = new_h, new_c

            output.append(h)

        output = torch.stack(output, dim=0)

        # Compute KL divergence
        kl = compute_KL(weights.view(-1), mean.view(-1), sigma.view(-1),
                        self.prior.sigma1, self.prior.sigma2, self.prior.pi_mixture)

        return output, (h, c), kl

    def forward(self, x, hidden):
        """
        Args:
            x: A (seq_len, batch, input_size) tensor containing input
                features.
            hidden: A tuple (h, c), which contains the hidden
                and cell state, where the size of both states is
                (num_layers, batch, hidden_size).

        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """

        self.kl = 0

        list_h = []
        list_c = []

        for layer_idx in range(self.num_layers):

            output, (h, c), kl = self.forward_layer(x, hidden, layer_idx)
            list_h.append(h)
            list_c.append(c)
            self.kl += kl

        h = torch.stack(list_h, dim=0)
        c = torch.stack(list_c, dim=0)
        hidden = (h, c)
        import ipdb; ipdb.set_trace()

        return output, hidden

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class BayesEmbedding(Module):

    def __init__(self, num_embeddings, embedding_dim, prior, init_scale):
        super(BayesEmbedding, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.prior = prior
        self.init_scale = init_scale
        self.max_norm = None
        self.norm_type = 2
        self.scale_grad_by_freq = False
        self.sparse = False
        self.padding_idx = -1

        emb_rho_min_init, emb_rho_max_init = prior.normal_init()
        mu, rho = get_bbb_variable([num_embeddings, embedding_dim],
                                   prior,
                                   init_scale,
                                   emb_rho_min_init,
                                   emb_rho_max_init)

        self.mu = mu
        self.rho = rho
        self.kl = None

    def forward(self, input):

        # Sample weight
        mean = self.mu
        sigma = F.softplus(self.rho) + 1E-5

        eps = Variable(torch.randn(mean.size()).type_as(mean.data))
        weights = mean + eps * sigma

        # Compute KL divergence
        self.kl = compute_KL(weights.view(-1), mean.view(-1), sigma.view(-1),
                             self.prior.sigma1, self.prior.sigma2, self.prior.pi_mixture)

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

    def __init__(self, in_features, out_features, prior, init_scale):
        super(BayesLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior = prior
        self.init_scale = init_scale

        sft_rho_min_init, sft_rho_max_init = prior.normal_init()

        mu, rho = get_bbb_variable((out_features, in_features),
                                   self.prior,
                                   self.init_scale,
                                   sft_rho_min_init,
                                   sft_rho_max_init)

        bias = Parameter(torch.Tensor(out_features))
        bias.data.fill_(0.)

        self.mu = mu
        self.rho = rho
        self.bias = bias
        self.kl = None

    def forward(self, input):

        # Sample weight
        mean = self.mu
        sigma = F.softplus(self.rho) + 1E-5

        # Sample weights from normal distribution
        eps = Variable(torch.randn(mean.size()).type_as(mean.data))
        weights = mean + eps * sigma

        # Compute KL divergence
        self.kl = compute_KL(weights.view(-1), mean.view(-1), sigma.view(-1),
                             self.prior.sigma1, self.prior.sigma2, self.prior.pi_mixture)

        return F.linear(input, weights, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'
