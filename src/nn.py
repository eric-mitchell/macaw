import torch
import torch.nn as nn
from typing import List, Callable, Optional


class DeepSet(nn.Module):
    def __init__(self, element_dim: int, encoding_dim: int = 256,
                 encoder_hidden: List[int] = [128, 128], output_dim: int = 128):
        super().__init__()
        encoder_hidden = [element_dim] + encoder_hidden + [encoding_dim]
        encoders = []
        for idx, (d, d_) in enumerate(zip(encoder_hidden[:-1], encoder_hidden[1:])):
            encoders.append(nn.Conv1d(d, d_, 1))
            if idx < len(encoder_hidden) - 1:
                encoders.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoders)
        
        self.aggregator = lambda x: x.mean(-1)

    def forward(self, x):
        encodings = self.encoder(x)
        return self.aggregator(encodings)


class CVAE(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, traj_dim: int, latent_dim: int = 32,
                 encoder_hidden: List[int] = [128, 64], prior_hidden: List[int] = [128, 64], decoder_hidden: List[int] = [128, 64],
                 condition_prior: bool = False, preprocess: bool = False):
        super().__init__()

        self._encoder = MLP([traj_dim] + encoder_hidden + [latent_dim * 2])
        if condition_prior:
            self._prior = MLP([observation_dim] + encoder_hidden + [latent_dim * 2])
        else:
            self._prior = None

        decoder_input_dim = latent_dim + observation_dim if not preprocess else latent_dim * 2
        self._decoder = MLP([decoder_input_dim] + decoder_hidden + [action_dim * 2])
        self._traj_encoder = DeepSet(observation_dim + action_dim, encoding_dim=traj_dim)

        if preprocess:
            self._state_preprocess = MLP([observation_dim] + decoder_hidden + [latent_dim])
            self._latent_preprocess = MLP([latent_dim] + decoder_hidden + [latent_dim])
            self._preprocess = lambda s, l: (self._state_preprocess(s), self._latent_preprocess(l))
        else:
            self._preprocess = None

        self._z = None
        self._latent_dim = latent_dim
        
    def sample(self, mu_logvar: torch.tensor):
        mu = mu_logvar[:,:mu_logvar.shape[-1] // 2]
        std = (mu_logvar[:,mu_logvar.shape[-1] // 2:] / 2).exp()
        return torch.empty_like(mu).normal_() * std + mu

    def fix(self, obs):
        self._z = self.prior(obs, True)[1].detach()

    def unfix(self):
        self._z = None

    def encode(self, traj: torch.tensor, sample: bool = False):
        traj_encoding = self._traj_encoder(traj)
        mu_logvar = self._encoder(traj_encoding)

        if sample:
            return mu_logvar, self.sample(mu_logvar)
        else:
            return mu_logvar

    def prior(self, obs: torch.tensor, sample: bool = False):
        if self._prior is None:
            d = self._encoder.seq[0].weight.device
            mu_logvar = torch.zeros(1, self._latent_dim * 2, device=d)
        else:
            mu_logvar = self._prior(obs)

        if sample:
            return mu_logvar, self.sample(mu_logvar)
        else:
            return mu_logvar

    def decode(self, latent: torch.tensor, obs: torch.tensor, sample: bool = False):
        if self._preprocess:
            obs, latent = self._preprocess(obs, latent)
        mu_logvar = self._decoder(torch.cat((latent, obs), -1))

        if sample:
            return mu_logvar, self.sample(mu_logvar)
        else:
            return mu_logvar

    def forward(self, obs: torch.tensor, traj: torch.tensor = None):
        if self._z is not None:
            z = self._z
        else:
            z = self.prior(obs, sample=True)[1]
        mu_logvar = self.decode(z, obs)
        return mu_logvar[:,:mu_logvar.shape[-1] // 2], (mu_logvar[:,mu_logvar.shape[-1] // 2:] / 2).exp()


class WLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias_size: Optional[int] = None, paaa=None):
        super().__init__()

        self.pa = paaa
        if bias_size is None:
            bias_size = out_features

        dim = 100
        self.z = nn.Parameter(torch.empty(dim).normal_(0, 1. / out_features))
        print(self.z.mean(), self.z.std().item())
        self.fc = nn.Linear(dim, in_features * out_features + out_features)
        self.seq = self.fc
        self.w_idx = in_features * out_features
        self.weight = self.fc.weight
        self._linear = self.fc
        self.out_f = out_features

    def adaptation_parameters(self):
        return [self.z]

    def forward(self, x: torch.tensor):
        #theta = self.fc(self.z + torch.empty_like(self.z).normal_(0, 1. / self.out_f))
        theta = self.fc(self.z)
        w = theta[:self.w_idx].view(x.shape[-1], -1)
        b = theta[self.w_idx:]
        return x @ w + b


class WLinearMix(nn.Module):
    def __init__(self, in_features: int, out_features: int, depth: int = 3, n_mix: int = 5, dim: int = 100):
        super().__init__()

        self.z = nn.Parameter(torch.empty(dim).normal_(0, 1. / (max(in_features, out_features) * n_mix)))
        print('Mix', self.z.mean(), self.z.std().item())

        self.m = nn.ModuleList([nn.Linear(dim, dim) for _ in range(depth - 1)])
        self.out = nn.Linear(dim, n_mix * (in_features * out_features + out_features))
        self.scales = nn.Parameter(torch.ones(n_mix))
        
        self.n_mix = n_mix
        self.w_idx = in_features * out_features
        self.weight = self.out.weight
        self._linear = self.out

    def forward(self, x: torch.tensor):
        theta = self.z
        for m in self.m:
            theta = m(theta).relu()
        theta = self.out(theta).tanh()
        theta = theta.view(-1, self.n_mix)# * self.scales.view(1,self.n_mix)
        w = theta[:self.w_idx].view(self.n_mix, -1, x.shape[-1])
        b = theta[self.w_idx:].view(self.n_mix, 1, -1)
        stack = (w.unsqueeze(1) @ x.unsqueeze(0).unsqueeze(-1)).squeeze(-1) + b
        #stack = torch.stack([x @ w[:,:,idx] + b[:,:,idx] for idx in range(self.n_mix)], -1)#torch.einsum('ij,klm->ilm', x,w) + b
        
        return stack.mean(0)


class BiasLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias_size: Optional[int] = None):
        super().__init__()
        if bias_size is None:
            bias_size = out_features

        self._linear = nn.Linear(in_features, out_features)
        self._bias = nn.Parameter(torch.empty_like(self._linear.bias).normal_(0, 1. / bias_size))
        self._weight = nn.Parameter(torch.empty(bias_size, out_features))
        nn.init.xavier_normal_(self._weight)

    def adaptation_parameters(self):
        return self.parameters()
        
    def forward(self, x: torch.tensor):
        return self._linear(x) + self._bias @ self._weight


class MLP(nn.Module):
    def __init__(self, layer_widths: List[int], final_activation: Callable = lambda x: x, bias_linear: bool = False, extra_head_layers: List[int] = None, w_linear: bool = False):
        super().__init__()

        if len(layer_widths) < 2:
            raise ValueError('Layer widths needs at least an in-dimension and out-dimension')

        self._final_activation = final_activation
        self.seq = nn.Sequential()
        self._head = extra_head_layers is not None

        if not w_linear:
            linear = BiasLinear if bias_linear else nn.Linear
        else:
            linear = WLinear
        self.bias_linear = bias_linear
        self.aparams = []
        
        for idx in range(len(layer_widths) - 1):
            w = linear(layer_widths[idx], layer_widths[idx + 1])
            self.aparams.extend(w.adaptation_parameters())
            self.seq.add_module(f'fc_{idx}', w)
            if idx < len(layer_widths) - 2:
                self.seq.add_module(f'relu_{idx}', nn.ReLU())

        if extra_head_layers is not None:
            self.pre_seq = self.seq[:-2]
            self.post_seq = self.seq[-2:]

            self.head_seq = nn.Sequential()
            extra_head_layers = [layer_widths[-2] + layer_widths[-1]] + extra_head_layers

            for idx, (infc, outfc) in enumerate(zip(extra_head_layers[:-1], extra_head_layers[1:])):
                self.head_seq.add_module(f'relu_{idx}', nn.ReLU())
                w = linear(extra_head_layers[idx], extra_head_layers[idx + 1])
                self.aparams.extend(w.adaptation_parameters())
                self.head_seq.add_module(f'fc_{idx}', w)

    def bias_parameters(self):
        return [self.seq[0]._linear.bias] if self.bias_linear else [self.seq[0].bias]

    def adaptation_parameters(self):
        return self.parameters()
        return self.aparams
    
    def forward(self, x: torch.tensor, acts: Optional[torch.tensor] = None):
        if self._head and acts is not None:
            h = self.pre_seq(x)
            head_input = torch.cat((h,acts), -1)
            return self._final_activation(self.post_seq(h)), self.head_seq(head_input)
        else:
            return self._final_activation(self.seq(x))
        

if __name__ == '__main__':
    mlp = MLP([1,5,8,2])
    x = torch.empty(10,1).normal_()
    print(mlp(x).shape)
