import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

#######
import curl
from curl.curl_sac import CURL
#######


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    print('sizes', sizes)
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    #######################
    def __init__(
        self, obs_dim, act_dim, hidden_sizes, activation,
        encoder_type, encoder_feature_dim, num_layers, num_filters):
        super().__init__()

        self.encoder = curl.encoder.make_encoder(
            encoder_type, obs_dim, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )
    #######################
    
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.

        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    
    def __init__(
        self, obs_dim, act_dim, hidden_sizes, activation,
        encoder_type, encoder_feature_dim, num_layers, num_filters):
        super().__init__(obs_dim, act_dim, hidden_sizes, activation,
        encoder_type, encoder_feature_dim, num_layers, num_filters)
        self.logits_net = mlp([encoder_feature_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        obs = self.encoder(obs)
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(
        self, obs_dim, act_dim, hidden_sizes, activation,
        encoder_type, encoder_feature_dim, num_layers, num_filters):
        super().__init__(obs_dim, act_dim, hidden_sizes, activation,
        encoder_type, encoder_feature_dim, num_layers, num_filters)
        print('act_dim', act_dim)
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([encoder_feature_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        print('obs.shape', obs.shape)
        obs = self.encoder(obs)
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.



class MLPActorCritic(nn.Module):


    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()
        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(observation_space, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(observation_space, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(observation_space, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]



class BROILActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, num_rew_fns, hidden_sizes=(64,64), activation=nn.Tanh, encoder_type='pixel', encoder_feature_dim=50, encoder_lr=1e-3, num_layers=4, num_filters=32, curl_latent_dim=128): ## encoder_type and everything after is from curl 
        super().__init__()
        print('action_space', act_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # new
        self.encoder_type = encoder_type
        self.curl_latent_dim = curl_latent_dim
        # \new

        # policy builder depends on action space
        if isinstance(act_dim, Box):
            print('box')
            self.pi = MLPGaussianActor(obs_dim, act_dim.shape[0], hidden_sizes, activation, encoder_type, encoder_feature_dim, num_layers, num_filters)
        elif isinstance(act_dim, Discrete):
            print('discrete')
            self.pi = MLPCategoricalActor(obs_dim, act_dim.n, hidden_sizes, activation,
        encoder_type, encoder_feature_dim, num_layers, num_filters)

        # build value function
        self.v  = BROILCritic(obs_dim, hidden_sizes, activation, num_rew_fns, encoder_type, encoder_feature_dim, num_layers, num_filters)
        if self.encoder_type == 'pixel':
            # create CURL encoder (the 128 batch size is probably unnecessary)
            self.CURL = CURL(obs_dim, encoder_feature_dim,
                        self.curl_latent_dim, self.v, output_type='continuous').to(self.device)

            # optimizer for critic encoder for reconstruction loss
            self.encoder_optimizer = torch.optim.Adam(
                self.v.encoder.parameters(), lr=encoder_lr
            )

            self.cpc_optimizer = torch.optim.Adam(
                self.CURL.parameters(), lr=encoder_lr
            )
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def step(self, obs, detach_encoder=False): ## added detach_encoder
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)

            if (type(v) == list):
                v = torch.from_numpy(np.asarray(v))

        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]

    def update_cpc(self, obs_anchor, obs_pos, cpc_kwargs, L, step):

        z_a = self.CURL.encode(obs_anchor)
        z_pos = self.CURL.encode(obs_pos, ema=True)

        logits = self.CURL.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.cross_entropy_loss(logits, labels)

        self.encoder_optimizer.zero_grad()
        self.cpc_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.cpc_optimizer.step()
        # if step % self.log_interval == 0:
        #    L.log('train/curl_loss', loss, step)


class BROILCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation, num_rew_fns, encoder_type, encoder_feature_dim, num_layers, num_filters):
        super().__init__()
        self.encoder = curl.encoder.make_encoder(
            encoder_type, obs_dim, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )
        self.encoder_momentum = curl.encoder.make_encoder(
            encoder_type, obs_dim, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )
        self.encoder_momentum.copy_conv_weights_from(self.encoder)
        self.v_nets = nn.ModuleList()
        for i in range(num_rew_fns):
            self.v_nets.append(mlp([encoder_feature_dim] + list(hidden_sizes) + [1], activation)) 

    def forward(self, obs):
        vals = []
        obs = self.encoder(obs)
        for v_net in self.v_nets:
            #v = torch.squeeze(v_net(obs), -1)
            v = v_net(obs)
            vals.append(v)
        val_tensor = torch.stack(vals, dim=1).squeeze()
        return val_tensor




