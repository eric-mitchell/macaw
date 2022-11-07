import argparse
import itertools
import json
import math
import os
import pickle
import random
import time
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import List, Optional

import higher
import numpy as np
import torch
import torch.autograd as A
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as O

warnings.filterwarnings("ignore", category=FutureWarning)
from torch.utils.tensorboard import SummaryWriter

from src.nn import CVAE, MLP
from src.utils import Experience, ReplayBuffer, RunningEstimator, argmax, kld


def env_action_dim(env):
    action_space = env.action_space.shape
    return action_space[0] if len(action_space) > 0 else 1


def print_(s: str, c: bool, end=None):
    if not c:
        if end is not None:
            print(s, end=end)
        else:
            print(s)


def DEBUG(s: str, c: bool):
    if c:
        print(s)


def check_config(config):
    """
    if len(config.train_buffer_paths):
        assert len(config.train_tasks) == len(config.train_buffer_paths), f'{len(config.train_tasks)}, {len(config.train_buffer_paths)}'
        assert len(config.test_tasks) == len(config.test_buffer_paths), f'{len(config.test_tasks)}, {len(config.test_buffers)}'
        if len(set(config.train_buffer_paths).intersection(set(config.test_buffer_paths))) > 0:
            print('WARNING: TEST AND TRAIN BUFFERS NOT DISJOINT')
    """
    if len(set(config.train_tasks).intersection(set(config.test_tasks))) > 0:
        print("WARNING: TEST AND TRAIN TASKS NOT DISJOINT")


class MACAW(object):
    def __init__(
        self,
        args: argparse.Namespace,
        task_config: dict,
        env,
        log_dir: str,
        name: str = None,
        training_iterations: int = 20000,
        visualization_interval: int = 100,
        silent: bool = False,
        instance_idx: int = 0,
        replay_buffer_length: int = 1000,
        gradient_steps_per_iteration: int = 1,
        discount_factor: float = 0.99,
        seed: int = 0,
    ):
        self._env = env
        self._log_dir = log_dir
        self._name = name if name is not None else "throwaway_test_run"
        self._args = args
        self._start_time = time.time()
        self.task_config = task_config
        self._instance_idx = instance_idx

        check_config(task_config)
        goal_dim = task_config.total_tasks if args.multitask else 0
        self._observation_dim = (
            env.observation_space.shape[0]
            + (args.trim_obs if args.trim_obs else 0)
            - goal_dim
        )
        self._action_dim = env_action_dim(env)

        policy_head = [32, 1] if args.advantage_head_coef is not None else None
        self._adaptation_policy = MLP(
            [self._observation_dim + goal_dim]
            + [args.net_width] * args.net_depth
            + [self._action_dim],
            final_activation=torch.tanh,
            bias_linear=not args.no_bias_linear,
            extra_head_layers=policy_head,
            w_linear=args.wlinear,
        ).to(args.device)

        if args.cvae:
            self._exploration_policy = CVAE(
                self._observation_dim + goal_dim,
                self._action_dim,
                args.latent_dim,
                condition_prior=args.cvae_prior_conditional,
                preprocess=args.cvae_preprocess,
            ).to(args.device)
        else:
            print("Using untrained exploration policy")
            self._exploration_policy = deepcopy(self._adaptation_policy)
            for p in self._exploration_policy.parameters():
                p.data = p.data.clone()

        self._q_function = MLP(
            [self._observation_dim + goal_dim + self._action_dim]
            + [args.net_width] * args.net_depth
            + [1],
            bias_linear=not args.no_bias_linear,
            w_linear=args.wlinear,
        ).to(args.device)
        self._value_function = MLP(
            [self._observation_dim + goal_dim]
            + [args.net_width] * args.net_depth
            + [1],
            bias_linear=not args.no_bias_linear,
            w_linear=args.wlinear,
        ).to(args.device)

        try:
            print(self._adaptation_policy.seq[0]._linear.weight.mean())
        except Exception as e:
            print(self._adaptation_policy.seq[0].weight.mean())

        print(
            f"Adaptation policy #params {torch.cat([p.view(-1) for p in self._adaptation_policy.parameters()]).shape[0]:,}"
        )
        print(
            f"Value function #params {torch.cat([p.view(-1) for p in self._value_function.parameters()]).shape[0]:,}"
        )

        self._adaptation_policy_optimizer = O.Adam(
            (
                self._adaptation_policy.parameters()
                if not args.multitask_bias_only
                else self._adaptation_policy.bias_parameters()
            ),
            lr=args.outer_policy_lr,
        )
        self._q_function_optimizer = O.Adam(
            self._q_function.parameters(), lr=args.outer_value_lr
        )
        self._value_function_optimizer = O.Adam(
            (
                self._value_function.parameters()
                if not args.multitask_bias_only
                else self._value_function.bias_parameters()
            ),
            lr=args.outer_value_lr,
        )
        if args.train_exploration or args.sample_exploration_inner:
            self._exploration_policy_optimizer = O.Adam(
                self._exploration_policy.parameters(), lr=args.exploration_lr
            )

        if args.archive is not None:
            archive_path = args.archive
            if self._instance_idx > 0:
                comps = args.archive.split("/")
                comps[-2] += f"_{instance_idx}"
                print("Remapping archive for new seed:")
                print(f"From:\t{args.archive}")
                archive_path = "/".join(comps)
                print(f"To:\t{archive_path}")

            print(f"Loading parameters from archive: {archive_path}")
            archive = torch.load(archive_path)
            self._value_function.load_state_dict(archive["vf"])
            self._adaptation_policy.load_state_dict(archive["policy"])
            self._value_function_optimizer.load_state_dict(archive["vf_opt"])
            self._adaptation_policy_optimizer.load_state_dict(archive["policy_opt"])
            self._policy_lrs = archive["policy_lrs"]
            self._value_lrs = archive["vf_lrs"]
            self._q_lrs = archive["q_lrs"]
            if "adv_coef" in archive:
                self._adv_coef = archive["adv_coef"]
            else:
                self._adv_coef = None
        else:
            self._policy_lrs = None
            self._value_lrs = None
            self._q_lrs = None
            self._adv_coef = None

        has_train_buffers = hasattr(task_config, "train_buffer_paths") and not args.eval
        has_test_buffers = hasattr(task_config, "test_buffer_paths")

        load_inner_buffers = has_train_buffers and args.load_inner_buffer
        load_outer_buffers = has_train_buffers and args.load_outer_buffer
        load_test_buffers = (
            has_test_buffers and args.load_inner_buffer
        )  # we want the test adaptation data the same as train

        inner_buffers = [
            task_config.train_buffer_paths.format(idx) if load_inner_buffers else None
            for idx in task_config.train_tasks
        ]
        outer_buffers = [
            task_config.train_buffer_paths.format(idx) if load_outer_buffers else None
            for idx in task_config.train_tasks
        ]
        test_buffers = [
            task_config.test_buffer_paths.format(idx) if load_test_buffers else None
            for idx in task_config.test_tasks
        ]

        self._test_buffers = [
            ReplayBuffer(
                args.inner_buffer_size,
                self._observation_dim,
                env_action_dim(self._env),
                discount_factor=discount_factor,
                immutable=test_buffers[i] is not None,
                load_from=test_buffers[i],
                silent=silent,
                skip=args.inner_buffer_skip,
                stream_to_disk=args.from_disk,
                mode=args.buffer_mode,
            )
            for i, task in enumerate(task_config.test_tasks)
        ]

        if not self._args.online_ft:
            self._inner_buffers = [
                ReplayBuffer(
                    args.inner_buffer_size,
                    self._observation_dim,
                    env_action_dim(self._env),
                    discount_factor=discount_factor,
                    immutable=args.offline or args.offline_inner,
                    load_from=inner_buffers[i],
                    silent=silent,
                    skip=args.inner_buffer_skip,
                    stream_to_disk=args.from_disk,
                    mode=args.buffer_mode,
                )
                for i, task in enumerate(task_config.train_tasks)
            ]

            if (
                args.offline
                and args.load_inner_buffer
                and args.load_outer_buffer
                and (args.replay_buffer_size == args.inner_buffer_size)
                and (args.buffer_skip == args.inner_buffer_skip)
                and args.buffer_mode == "end"
            ):
                self._outer_buffers = self._inner_buffers
            else:
                self._outer_buffers = [
                    ReplayBuffer(
                        args.replay_buffer_size,
                        self._observation_dim,
                        env_action_dim(self._env),
                        discount_factor=discount_factor,
                        immutable=args.offline or args.offline_outer,
                        load_from=outer_buffers[i],
                        silent=silent,
                        skip=args.buffer_skip,
                        stream_to_disk=args.from_disk,
                    )
                    for i, task in enumerate(task_config.train_tasks)
                ]

        self._training_iterations = training_iterations
        if self._policy_lrs is None:
            self._policy_lrs = [
                torch.nn.Parameter(
                    torch.tensor(
                        float(np.log(self._args.inner_policy_lr))
                        if not self._args.multitask
                        else 10000.0
                    ).to(args.device)
                )
                for p in self._adaptation_policy.adaptation_parameters()
            ]
            self._value_lrs = [
                torch.nn.Parameter(
                    torch.tensor(
                        float(np.log(self._args.inner_value_lr))
                        if not self._args.multitask
                        else 10000.0
                    ).to(args.device)
                )
                for p in self._value_function.adaptation_parameters()
            ]
            self._q_lrs = [
                torch.nn.Parameter(
                    torch.tensor(
                        float(np.log(self._args.inner_value_lr))
                        if not self._args.multitask
                        else 10000.0
                    ).to(args.device)
                )
                for p in self._q_function.adaptation_parameters()
            ]
            if args.advantage_head_coef is not None:
                self._adv_coef = torch.nn.Parameter(
                    torch.tensor(float(np.log(args.advantage_head_coef))).to(
                        args.device
                    )
                )

        self._policy_lr_optimizer = O.Adam(self._policy_lrs, lr=self._args.lrlr)
        self._value_lr_optimizer = O.Adam(self._value_lrs, lr=self._args.lrlr)
        self._q_lr_optimizer = O.Adam(self._q_lrs, lr=self._args.lrlr)
        if args.advantage_head_coef is not None:
            self._adv_coef_optimizer = O.Adam([self._adv_coef], lr=self._args.lrlr)

        self._adaptation_temperature = args.adaptation_temp
        self._device = torch.device(args.device)
        self._cpu = torch.device("cpu")
        self._advantage_clamp = np.log(args.exp_advantage_clip)
        self._action_sigma = args.action_sigma
        self._visualization_interval = visualization_interval
        self._silent = silent
        self._gradient_steps_per_iteration = gradient_steps_per_iteration
        self._grad_clip = args.grad_clip
        self._env_seeds = np.random.randint(1e10, size=(int(1e7),))
        self._rollout_counter = 0
        self._value_estimators = [RunningEstimator() for _ in self._env.tasks]
        self._q_estimators = [RunningEstimator() for _ in self._env.tasks]
        self._maml_steps = args.maml_steps
        self._max_maml_steps = args.maml_steps

    #################################################################
    ################# SUBROUTINES FOR TRAINING ######################
    #################################################################
    # @profile
    def _rollout_policy(
        self,
        policy: MLP,
        env,
        sample_mode: bool = False,
        random: bool = False,
        render: bool = False,
    ) -> List[Experience]:
        env.seed(self._env_seeds[self._rollout_counter].item())
        self._rollout_counter += 1
        trajectory = []
        state = env.reset()
        if self._args.trim_obs is not None:
            state = np.concatenate((state, np.zeros((self._args.trim_obs,))))
        if render:
            env.render()
        done = False
        total_reward = 0
        episode_t = 0

        if isinstance(policy, CVAE):
            policy.fix(
                torch.tensor(state, device=self._args.device).unsqueeze(0).float()
            )

        success = False
        policy.eval()
        while not done:
            if self._args.multitask and sample_mode:
                state[-self.task_config.total_tasks :] = 0
            if not random:
                with torch.no_grad():
                    action_sigma = self._action_sigma
                    if isinstance(policy, CVAE):
                        mu, action_sigma = policy(
                            torch.tensor(state, device=self._args.device)
                            .unsqueeze(0)
                            .float()
                        )
                        mu = mu.squeeze()
                        action_sigma = action_sigma.squeeze().clamp(max=0.5)
                    else:
                        mu = policy(
                            torch.tensor(state, device=self._args.device)
                            .unsqueeze(0)
                            .float()
                        ).squeeze()

                    if sample_mode:
                        action = mu
                    else:
                        action = mu + torch.empty_like(mu).normal_() * action_sigma

                    log_prob = (
                        D.Normal(mu, torch.empty_like(mu).fill_(self._action_sigma))
                        .log_prob(action)
                        .to(self._cpu)
                        .numpy()
                        .sum()
                    )
                    action = (
                        action.squeeze()
                        .to(self._cpu)
                        .numpy()
                        .clip(min=env.action_space.low, max=env.action_space.high)
                    )
            else:
                action = env.action_space.sample()
                log_prob = math.log(1 / 2 ** env.action_space.shape[0])

            next_state, reward, done, info_dict = env.step(action)
            if self._args.trim_obs is not None:
                next_state = np.concatenate(
                    (next_state, np.zeros((self._args.trim_obs,)))
                )

            if "success" in info_dict and info_dict["success"]:
                success = True

            if render:
                env.render()
            trajectory.append(Experience(state, action, next_state, reward, done))
            state = next_state
            total_reward += reward
            episode_t += 1
            if episode_t >= env._max_episode_steps or done:
                break

        if isinstance(policy, CVAE):
            policy.unfix()
        return trajectory, total_reward, success

    def add_task_description(self, obs, task_idx: int):
        if not self._args.multitask:
            return obs

        idx = torch.zeros((obs.shape[0], self.task_config.total_tasks)).to(obs.device)
        if task_idx is not None:
            idx[:, task_idx] = 1
        return torch.cat((obs, idx), -1)

    # @profile
    def mc_value_estimates_on_batch(
        self, value_function, batch, task_idx, no_bootstrap=False
    ):
        mc_value_estimates = batch[:, -1:]
        if not no_bootstrap:
            terminal_state_value_estimates = value_function(
                self.add_task_description(
                    batch[
                        :,
                        self._observation_dim * 2
                        + self._action_dim : self._observation_dim * 3
                        + self._action_dim,
                    ],
                    task_idx,
                )
            )
            bootstrap_correction = (
                batch[:, -4:-3] * terminal_state_value_estimates
            )  # I know this magic number indexing is heinous... I'm sorry
            mc_value_estimates = mc_value_estimates + bootstrap_correction

        return mc_value_estimates

    # @profile
    def q_function_loss_on_batch(
        self,
        q_function,
        value_function,
        batch,
        inner: bool = False,
        task_idx: int = None,
    ):
        q_estimates = q_function(
            self.add_task_description(
                torch.cat(
                    (
                        batch[:, : self._observation_dim],
                        batch[
                            :,
                            self._observation_dim : self._observation_dim
                            + self._action_dim,
                        ],
                    ),
                    -1,
                ),
                task_idx,
            )
        )
        with torch.no_grad():
            mc_value_estimates = self.mc_value_estimates_on_batch(
                value_function,
                batch,
                task_idx,
                self._args.no_bootstrap if inner else False,
            )

        return (q_estimates - mc_value_estimates).pow(2).mean()

    # @profile
    def value_function_loss_on_batch(
        self,
        value_function,
        batch,
        inner: bool = False,
        task_idx: int = None,
        iweights: torch.tensor = None,
        target=None,
    ):
        value_estimates = value_function(
            self.add_task_description(batch[:, : self._observation_dim], task_idx)
        )
        with torch.no_grad():
            if target is None:
                target = value_function
            mc_value_estimates = self.mc_value_estimates_on_batch(
                target,
                batch,
                task_idx,
                self._args.no_bootstrap and (inner or self._args.multitask),
            )

            targets = mc_value_estimates
            if self._args.log_targets:
                targets[torch.logical_and(targets > -1, targets < 1)] = 0
                targets[targets > 1] = targets[targets > 1].log()
                targets[targets < -1] = -targets[targets < -1].abs().log()
                targets = targets.clone()

        DEBUG(
            f"({task_idx}) VALUE: {value_estimates.abs().mean()}, {targets.abs().mean()}",
            self._args.debug,
        )
        if self._args.huber and not inner:
            losses = F.smooth_l1_loss(value_estimates, targets, reduction="none")
        else:
            losses = (value_estimates - targets).pow(2)

        return (
            losses.mean(),
            value_estimates.mean(),
            mc_value_estimates.mean(),
            mc_value_estimates.std(),
        )

    def imitation_loss_on_batch(
        self, policy, batch, task_idx: int, inner: bool = False
    ):
        original_action = batch[
            :, self._observation_dim : self._observation_dim + self._action_dim
        ]
        action_mu = policy(
            self.add_task_description(batch[:, : self._observation_dim], task_idx)
        )
        action_sigma = torch.empty_like(action_mu).fill_(self._action_sigma)

        action_distribution = D.Normal(action_mu, action_sigma)
        action_log_probs = action_distribution.log_prob(original_action).sum(-1)

        losses = -action_log_probs

        return losses.mean()

    # @profile
    def adaptation_policy_loss_on_batch(
        self,
        policy,
        q_function,
        value_function,
        batch,
        task_idx: int,
        inner: bool = False,
        iweights: torch.tensor = None,
        online: bool = False,
    ):
        with torch.no_grad():
            value_estimates = value_function(
                self.add_task_description(batch[:, : self._observation_dim], task_idx)
            )
            if q_function is not None:
                action_value_estimates = q_function(
                    torch.cat(
                        (
                            batch[:, : self._observation_dim],
                            batch[
                                :,
                                self._observation_dim : self._observation_dim
                                + self._action_dim,
                            ],
                        ),
                        -1,
                    )
                )
            else:
                action_value_estimates = self.mc_value_estimates_on_batch(
                    value_function, batch, task_idx
                )

            advantages = (action_value_estimates - value_estimates).squeeze(-1)
            if self._args.no_norm:
                weights = advantages.clamp(
                    min=-self._advantage_clamp, max=self._advantage_clamp
                ).exp()
            else:
                normalized_advantages = (
                    (1 / self._adaptation_temperature)
                    * (advantages - advantages.mean())
                    / advantages.std()
                )
                weights = normalized_advantages.clamp(max=self._advantage_clamp).exp()
            DEBUG(
                f"POLICY {advantages.abs().mean()}, {weights.abs().mean()}",
                self._args.debug,
            )

        original_action = batch[
            :, self._observation_dim : self._observation_dim + self._action_dim
        ]
        if self._args.advantage_head_coef is not None:
            action_mu, advantage_prediction = policy(
                self.add_task_description(batch[:, : self._observation_dim], task_idx),
                batch[
                    :, self._observation_dim : self._observation_dim + self._action_dim
                ],
            )
        else:
            action_mu = policy(
                self.add_task_description(batch[:, : self._observation_dim], task_idx)
            )
        action_sigma = torch.empty_like(action_mu).fill_(self._action_sigma)
        action_distribution = D.Normal(action_mu, action_sigma)
        action_log_probs = action_distribution.log_prob(
            batch[:, self._observation_dim : self._observation_dim + self._action_dim]
        ).sum(-1)

        losses = -(action_log_probs * weights)

        if iweights is not None:
            losses = losses * iweights

        adv_prediction_loss = None
        if inner:
            if self._args.advantage_head_coef is not None:
                adv_prediction_loss = (
                    F.softplus(self._adv_coef)
                    * (advantage_prediction.squeeze() - advantages) ** 2
                )
                losses = losses + adv_prediction_loss
                adv_prediction_loss = adv_prediction_loss.mean()

        return losses.mean(), advantages.mean(), weights, adv_prediction_loss

    def update_model(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        clip: float = None,
        extra_grad: list = None,
    ):
        if clip is not None:
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        else:
            grad = None

        optimizer.step()
        optimizer.zero_grad()

        return grad

    def update_params(
        self,
        params: list,
        optimizer: torch.optim.Optimizer,
        clip: float = None,
        extra_grad: list = None,
    ):
        optimizer.step()
        optimizer.zero_grad()

    def soft_update(self, source, target):
        for param_source, param_target in zip(
            source.named_parameters(), target.named_parameters()
        ):
            assert param_source[0] == param_target[0]
            param_target[1].data = (
                self._args.target_vf_alpha * param_target[1].data
                + (1 - self._args.target_vf_alpha) * param_source[1].data
            )

    def eval_multitask(self, train_step_idx: int, writer: SummaryWriter):
        rewards = np.full(
            (len(self.task_config.test_tasks), self._args.eval_maml_steps + 1),
            float("nan"),
        )
        trajectories, successes = [], []

        log_steps = [1, 5, 20]
        reward_dict = defaultdict(list)
        success_dict = defaultdict(list)
        for i, (test_task_idx, test_buffer) in enumerate(
            zip(self.task_config.test_tasks, self._test_buffers)
        ):
            self._env.set_task_idx(test_task_idx)

            if self._args.eval:
                adapted_trajectory, adapted_reward, success = self._rollout_policy(
                    self._adaptation_policy,
                    self._env,
                    sample_mode=True,
                    render=self._args.render,
                )
                trajectories.append(adapted_trajectory)
                rewards[i, 0] = adapted_reward
                successes.append(success)
                writer.add_scalar(
                    f"Eval_Reward/Task_{test_task_idx}", adapted_reward, 0
                )

            vf = deepcopy(self._value_function)
            ap = deepcopy(self._adaptation_policy)
            opt = O.Adam(vf.parameters(), lr=self._args.inner_value_lr)
            ap_opt = O.Adam(ap.parameters(), lr=self._args.inner_policy_lr)
            batch = torch.tensor(
                test_buffer.sample(self._args.eval_batch_size), requires_grad=False
            ).to(self._device)
            for step in range(max(log_steps)):
                vf_loss, _, _, _ = self.value_function_loss_on_batch(
                    vf, batch, task_idx=None, inner=True
                )
                vf_loss.backward()
                opt.step()
                opt.zero_grad()

                ap_loss, _, _, _ = self.adaptation_policy_loss_on_batch(
                    ap, None, vf, batch, task_idx=None, inner=True
                )
                ap_loss.backward()
                ap_opt.step()
                ap_opt.zero_grad()

                adapted_trajectory, adapted_reward, success = self._rollout_policy(
                    ap, self._env, sample_mode=True
                )
                if (step + 1) in log_steps:
                    reward_dict[step + 1].append(adapted_reward)
                    success_dict[step + 1].append(success)
                    writer.add_scalar(
                        f"FT_Eval_Reward/Task_{i}_Step{step}",
                        adapted_reward,
                        train_step_idx,
                    )
                    writer.add_scalar(
                        f"FT_Eval_Success/Task_{i}_Step{step}",
                        int(success),
                        train_step_idx,
                    )
                if self._args.eval:
                    rewards[i, step + 1] = adapted_reward
                    writer.add_scalar(
                        f"Eval_Reward/Task_{test_task_idx}", adapted_reward, step + 1
                    )
            trajectories.append(adapted_trajectory)
        for s in log_steps:
            writer.add_scalar(
                f"FT_Eval_Reward/Mean_Step{s}", np.mean(reward_dict[s]), train_step_idx
            )
            writer.add_scalar(
                f"FT_Eval_Success/Mean_Step{s}",
                np.mean(success_dict[s]),
                train_step_idx,
            )

        if not self._args.eval:
            rewards = np.array(reward_dict[log_steps[-1]])[:, None]
        if self._args.eval:
            for idx, r in enumerate(rewards.mean(0)):
                writer.add_scalar(f"Eval_Reward/Mean", r, idx)
        else:
            writer.add_scalar(f"Eval_Reward/Mean", rewards.mean(0)[-1], train_step_idx)
        return trajectories, rewards[:, -1], np.array(successes)

    def eval_macaw(
        self,
        train_step_idx: int,
        writer: SummaryWriter,
        ft: str = "offline",
        ft_steps: int = 19,
        burn_in: int = 25,
        steps_per_rollout: int = 1,
        log_path: str = None,
    ):
        if ft == "offline":
            ft_steps = 0
            burn_in = 0
        else:
            print(
                f"Beginning fine-tuning for {ft_steps} steps, including {burn_in} of burn in."
            )
        trajectories, successes = [], []
        rewards = np.full(
            (len(self.task_config.test_tasks), 2 + ft_steps), float("nan")
        )
        successes = np.full(
            (len(self.task_config.test_tasks), 2 + ft_steps), float("nan")
        )
        for i, (test_task_idx, test_buffer) in enumerate(
            zip(self.task_config.test_tasks[::-1], self._test_buffers[::-1])
        ):
            self._env.set_task_idx(test_task_idx)
            if ft == "online":
                print(f"Beginning fine-tuning on task {test_task_idx}")
                filepath = log_path + "/rewards"
                if log_path is not None:
                    print(f"Saving results to {filepath}")

            adapted_trajectory, adapted_reward, success = self._rollout_policy(
                self._adaptation_policy,
                self._env,
                sample_mode=True,
                render=self._args.render,
            )
            trajectories.append(adapted_trajectory)
            rewards[i, 0] = adapted_reward
            successes[i, 0] = success

            value_batch, value_batch_dict = test_buffer.sample(
                self._args.eval_batch_size, return_both=True
            )
            value_batch = torch.tensor(value_batch, requires_grad=False).to(
                self._device
            )
            policy_batch, policy_batch_dict = value_batch, value_batch_dict

            value_function = deepcopy(self._value_function)
            vf_target = deepcopy(value_function)
            DEBUG(
                "******************************************* EVAL **********************************",
                self._args.debug,
            )
            opt = O.SGD(
                [
                    {"params": p, "lr": None}
                    for p in value_function.adaptation_parameters()
                ]
            )
            with higher.innerloop_ctx(
                value_function,
                opt,
                override={"lr": [F.softplus(l) for l in self._value_lrs]},
            ) as (f_value_function, diff_value_opt):
                if not self._args.imitation:
                    loss, _, _, _ = self.value_function_loss_on_batch(
                        f_value_function,
                        value_batch,
                        task_idx=test_task_idx,
                        inner=True,
                        target=vf_target,
                    )
                    diff_value_opt.step(loss)

                    # Soft update target value function parameters
                    self.soft_update(f_value_function, vf_target)

                policy = deepcopy(self._adaptation_policy)
                policy_opt = O.SGD(
                    [{"params": p, "lr": None} for p in policy.adaptation_parameters()]
                )
                with higher.innerloop_ctx(
                    policy,
                    policy_opt,
                    override={"lr": [F.softplus(l) for l in self._policy_lrs]},
                ) as (f_policy, diff_policy_opt):
                    if self._args.imitation:
                        loss = self.imitation_loss_on_batch(
                            f_policy, policy_batch, None, inner=True
                        )
                    else:
                        loss, _, _, _ = self.adaptation_policy_loss_on_batch(
                            f_policy,
                            None,
                            f_value_function,
                            policy_batch,
                            test_task_idx,
                            inner=True,
                        )
                    diff_policy_opt.step(loss)

                    adapted_trajectory, adapted_reward, success = self._rollout_policy(
                        f_policy, self._env, sample_mode=True, render=self._args.render
                    )
                    trajectories.append(adapted_trajectory)
                    rewards[i, 1] = adapted_reward
                    successes[i, 1] = success

                # After one offline adaptation step, either do some more offline adaptation, or run online adaptation
                adapted_vf = deepcopy(self._value_function)
                for targ, src in zip(
                    adapted_vf.parameters(), f_value_function.parameters()
                ):
                    targ.data[:] = src.data[:]
                adapted_policy = deepcopy(self._adaptation_policy)
                for targ, src in zip(
                    adapted_policy.parameters(), f_policy.parameters()
                ):
                    targ.data[:] = src.data[:]

                del f_policy, diff_policy_opt
                del f_value_function, diff_value_opt

                if not self._args.imitation and ft != "offline":
                    vf_target = deepcopy(adapted_vf)
                    ft_v_opt = O.Adam(adapted_vf.parameters(), lr=1e-5)
                    ft_p_opt = O.Adam(adapted_policy.parameters(), lr=1e-5)
                    buf_size = 50000  # self._env._max_episode_steps * ft_steps
                    replay_buffer = ReplayBuffer.from_dict(
                        buf_size, value_batch_dict, self._silent
                    )
                    replay_buffer._stored_steps = 0
                    replay_buffer._write_location = 0

                    start = None
                    for eval_step in range(ft_steps):
                        if eval_step == burn_in or eval_step == 0:
                            start = time.time()
                        if eval_step > 0:
                            step_time = (time.time() - start) / max(
                                1, (eval_step - burn_in)
                            )
                            print(
                                self._instance_idx,
                                eval_step,
                                eval_rewards,
                                adapted_reward,
                                step_time,
                                (ft_steps - eval_step) * step_time / 3600,
                            )

                        if ft == "online":
                            writer.add_scalar(
                                f"Eval_Buf_Size/Task_{test_task_idx}",
                                len(replay_buffer),
                                eval_step,
                            )
                        eval_rewards, eval_successes = [], []
                        for traj_idx in range(3):
                            _, adapted_reward, success = self._rollout_policy(
                                adapted_policy,
                                self._env,
                                sample_mode=True,
                                render=self._args.render,
                            )
                            eval_rewards.append(adapted_reward)
                            eval_successes.append(success)
                        trajectories.append(adapted_trajectory)
                        adapted_reward = sum(eval_rewards) / len(eval_rewards)
                        adapted_success = sum(eval_successes) / len(eval_successes)
                        rewards[i, 2 + eval_step] = adapted_reward
                        successes[i, 2 + eval_step] = success
                        if ft == "online":
                            noise_trajectory, _, _ = self._rollout_policy(
                                adapted_policy,
                                self._env,
                                sample_mode=False,
                                render=self._args.render,
                            )
                            replay_buffer.add_trajectory(noise_trajectory)
                            writer.add_scalar(
                                f"Eval_Reward_FTStep/Task_{test_task_idx}",
                                adapted_reward,
                                eval_step,
                            )
                            writer.add_scalar(
                                f"Eval_Success_FTStep/Task_{test_task_idx}",
                                success,
                                eval_step,
                            )

                        if replay_buffer._stored_steps >= self._args.eval_batch_size:
                            if eval_step > burn_in:
                                for _ in range(steps_per_rollout):
                                    ft_value_batch = torch.tensor(
                                        replay_buffer.sample(
                                            self._args.eval_batch_size
                                        ),
                                        requires_grad=False,
                                    ).to(self._device)

                                    loss, _, _, _ = self.value_function_loss_on_batch(
                                        adapted_vf,
                                        ft_value_batch,
                                        task_idx=test_task_idx,
                                        inner=True,
                                        target=None,
                                    )
                                    loss.backward()
                                    ft_v_opt.step()
                                    ft_v_opt.zero_grad()
                                if ft == "online":
                                    writer.add_scalar(
                                        f"FTStep_Value_Loss/Task_{test_task_idx}",
                                        loss.item(),
                                        eval_step,
                                    )

                            if eval_step > burn_in:
                                for _ in range(steps_per_rollout * 2):
                                    ft_policy_batch = torch.tensor(
                                        replay_buffer.sample(
                                            self._args.eval_batch_size
                                        ),
                                        requires_grad=False,
                                    ).to(self._device)
                                    if self._args.imitation:
                                        policy_loss = self.imitation_loss_on_batch(
                                            adapted_policy,
                                            ft_policy_batch,
                                            None,
                                            inner=True,
                                        )
                                    else:
                                        (
                                            policy_loss,
                                            _,
                                            _,
                                            _,
                                        ) = self.adaptation_policy_loss_on_batch(
                                            adapted_policy,
                                            None,
                                            adapted_vf,
                                            ft_policy_batch,
                                            test_task_idx,
                                        )
                                    policy_loss.backward()
                                    ft_p_opt.step()
                                    ft_p_opt.zero_grad()
                                if ft == "online":
                                    writer.add_scalar(
                                        f"FTStep_Policy_Loss/Task_{test_task_idx}",
                                        policy_loss.item(),
                                        eval_step,
                                    )

                writer.add_scalar(
                    f"Eval_Reward/Task_{test_task_idx}", rewards[i, 1], train_step_idx
                )
                writer.add_scalar(
                    f"Eval_Reward_FT/Task_{test_task_idx}",
                    rewards[i, -1],
                    train_step_idx,
                )

                writer.add_scalar(
                    f"Eval_Success/Task_{test_task_idx}",
                    successes[i, 1],
                    train_step_idx,
                )
                writer.add_scalar(
                    f"Eval_Success_FT/Task_{test_task_idx}",
                    successes[i, -1],
                    train_step_idx,
                )

        if log_path is not None:
            np.save(filepath, rewards)

        writer.add_scalar(f"Eval_Reward/Mean", rewards.mean(0)[1], train_step_idx)
        writer.add_scalar(f"Eval_Success/Mean", successes.mean(0)[1], train_step_idx)
        writer.add_scalar(f"Eval_Reward_FT/Mean", rewards.mean(0)[-1], train_step_idx)
        writer.add_scalar(
            f"Eval_Success_FT/Mean", successes.mean(0)[-1], train_step_idx
        )
        mean_rewards = rewards.mean(0)
        mean_successes = successes.mean(0)
        for rollout_idx in range(ft_steps):
            writer.add_scalar(
                f"Eval_Reward_FTStep/Mean", mean_rewards[rollout_idx], rollout_idx
            )
            writer.add_scalar(
                f"Eval_Success_FTStep/Mean", mean_successes[rollout_idx], rollout_idx
            )

        return trajectories, rewards[:, -1], successes

    def eval(self, train_step_idx: int, writer: SummaryWriter):
        if self._args.multitask:
            return self.eval_multitask(train_step_idx, writer)
        else:
            return self.eval_macaw(train_step_idx, writer)

    # This function is the body of the main training loop [L4]
    # At every iteration, it adds rollouts from the exploration policy and one of the adapted policies
    #  to the replay buffer. It also updates the adaptation value function, adaptation policy, and
    #  exploration policy
    # @profile
    def train_step(self, train_step_idx: int, writer: Optional[SummaryWriter] = None):
        if train_step_idx % self._visualization_interval == 0:
            test_rollouts, test_rewards, successes = self.eval(train_step_idx, writer)
        else:
            test_rollouts = []
            test_rewards = []
            successes = []

        if self._args.eval:
            return (
                test_rollouts,
                test_rewards,
                test_rewards,
                [0],
                [0],
                self._value_function,
                successes,
            )

        q_functions = []
        meta_q_grads = []
        meta_value_grads = []
        meta_policy_grads = []
        exploration_grads = []
        train_rewards = []
        rollouts = []
        successes = []
        if (
            self._args.task_batch_size is not None
            and len(self.task_config.train_tasks) > self._args.task_batch_size
        ):
            tasks = random.sample(
                self.task_config.train_tasks, self._args.task_batch_size
            )
        else:
            tasks = self.task_config.train_tasks

        for i, (train_task_idx, inner_buffer, outer_buffer) in enumerate(
            zip(self.task_config.train_tasks, self._inner_buffers, self._outer_buffers)
        ):
            DEBUG(
                f"**************** TASK IDX {train_task_idx} ***********",
                self._args.debug,
            )

            # Only train on the randomly selected tasks for this iteration
            if train_task_idx not in tasks:
                continue

            self._env.set_task_idx(train_task_idx)

            # Sample J training batches for independent adaptations [L7]
            value_batch = torch.tensor(
                inner_buffer.sample(
                    self._args.inner_batch_size, contiguous=self._args.contiguous
                ),
                requires_grad=False,
            ).to(self._device)
            policy_batch = value_batch
            meta_batch = torch.tensor(
                outer_buffer.sample(self._args.batch_size), requires_grad=False
            ).to(self._device)
            policy_meta_batch = meta_batch

            inner_value_losses = []
            meta_value_losses = []
            inner_policy_losses = []
            adv_policy_losses = []
            meta_policy_losses = []
            value_lr_grads = []
            policy_lr_grads = []
            inner_mc_means, inner_mc_stds = [], []
            outer_mc_means, outer_mc_stds = [], []
            inner_values, outer_values = [], []
            inner_weights, outer_weights = [], []
            inner_advantages, outer_advantages = [], []

            iweights_ = None
            iweights_no_action_ = None

            ##################################################################################################
            # Adapt value function and collect meta-gradients
            ##################################################################################################
            if self._args.multitask:
                vf_target = self._value_function
                (
                    meta_value_function_loss,
                    value,
                    mc,
                    mc_std,
                ) = self.value_function_loss_on_batch(
                    self._value_function,
                    meta_batch,
                    task_idx=train_task_idx,
                    target=vf_target,
                )
                total_vf_loss = meta_value_function_loss / len(
                    self.task_config.train_tasks
                )
                total_vf_loss.backward()

                outer_values.append(value.item())
                outer_mc_means.append(mc.item())
                outer_mc_stds.append(mc_std.item())
                meta_value_losses.append(meta_value_function_loss.item())

                (
                    meta_policy_loss,
                    outer_adv,
                    outer_weights_,
                    _,
                ) = self.adaptation_policy_loss_on_batch(
                    self._adaptation_policy,
                    None,
                    self._value_function,
                    policy_meta_batch,
                    train_task_idx,
                )
                (meta_policy_loss / len(self.task_config.train_tasks)).backward()

                outer_weights.append(outer_weights_.mean().item())
                outer_advantages.append(outer_adv.item())
                meta_policy_losses.append(meta_policy_loss.item())

                # Sample adapted policy trajectory, add to replay buffer i [L12]
                if train_step_idx % self._gradient_steps_per_iteration == 0:
                    adapted_trajectory, adapted_reward, success = self._rollout_policy(
                        self._adaptation_policy,
                        self._env,
                        sample_mode=self._args.offline,
                    )
                    train_rewards.append(adapted_reward)
                    successes.append(success)

                    if not (self._args.offline or self._args.offline_inner):
                        inner_buffer.add_trajectory(adapted_trajectory)
                    if not (self._args.offline or self._args.offline_outer):
                        outer_buffer.add_trajectory(adapted_trajectory)
                else:
                    success = False
            else:
                if True:
                    if not self._args.imitation:
                        vf = self._value_function
                        vf.train()
                        vf_target = deepcopy(vf)
                        opt = O.SGD(
                            [
                                {"params": p, "lr": None}
                                for p in vf.adaptation_parameters()
                            ]
                        )
                        with higher.innerloop_ctx(
                            vf,
                            opt,
                            override={"lr": [F.softplus(l) for l in self._value_lrs]},
                            copy_initial_weights=False,
                        ) as (f_value_function, diff_value_opt):
                            if len(self._env.tasks) > 1:
                                for step in range(self._maml_steps):
                                    DEBUG(
                                        f"################# VALUE STEP {step} ###################",
                                        self._args.debug,
                                    )
                                    sub_batch = value_batch.view(
                                        self._args.maml_steps,
                                        value_batch.shape[0] // self._args.maml_steps,
                                        *value_batch.shape[1:],
                                    )[step]
                                    (
                                        loss,
                                        value_inner,
                                        mc_inner,
                                        mc_std_inner,
                                    ) = self.value_function_loss_on_batch(
                                        f_value_function,
                                        sub_batch,
                                        inner=True,
                                        task_idx=train_task_idx,
                                        target=vf_target,
                                    )  # , iweights=iweights_no_action_)

                                    inner_values.append(value_inner.item())
                                    inner_mc_means.append(mc_inner.item())
                                    inner_mc_stds.append(mc_std_inner.item())
                                    diff_value_opt.step(loss)
                                    inner_value_losses.append(loss.item())

                                    # Soft update target value function parameters
                                    self.soft_update(f_value_function, vf_target)

                            # Collect grads for the value function update in the outer loop [L14],
                            #  which is not actually performed here
                            (
                                meta_value_function_loss,
                                value,
                                mc,
                                mc_std,
                            ) = self.value_function_loss_on_batch(
                                f_value_function,
                                meta_batch,
                                task_idx=train_task_idx,
                                target=vf_target,
                            )
                            total_vf_loss = meta_value_function_loss / len(
                                self.task_config.train_tasks
                            )
                            if self._args.value_reg > 0:
                                total_vf_loss = (
                                    total_vf_loss
                                    + self._args.value_reg
                                    * self._value_function(
                                        value_batch[:, : self._observation_dim]
                                    )
                                    .pow(2)
                                    .mean()
                                )
                            total_vf_loss.backward()

                            outer_values.append(value.item())
                            outer_mc_means.append(mc.item())
                            outer_mc_stds.append(mc_std.item())
                            meta_value_losses.append(meta_value_function_loss.item())
                            ##################################################################################################

                        ##################################################################################################
                        # Adapt policy and collect meta-gradients
                        ##################################################################################################
                        adapted_value_function = f_value_function
                        adapted_q_function = f_q_function if self._args.q else None

                    opt = O.SGD(
                        [
                            {"params": p, "lr": None}
                            for p in self._adaptation_policy.adaptation_parameters()
                        ]
                    )
                    self._adaptation_policy.train()
                    with higher.innerloop_ctx(
                        self._adaptation_policy,
                        opt,
                        override={"lr": [F.softplus(l) for l in self._policy_lrs]},
                        copy_initial_weights=False,
                    ) as (f_adaptation_policy, diff_policy_opt):
                        if len(self._env.tasks) > 1:
                            for step in range(self._maml_steps):
                                DEBUG(
                                    f"################# POLICY STEP {step} ###################",
                                    self._args.debug,
                                )
                                sub_batch = policy_batch.view(
                                    self._args.maml_steps,
                                    policy_batch.shape[0] // self._args.maml_steps,
                                    *policy_batch.shape[1:],
                                )[step]
                                if self._args.imitation:
                                    loss = self.imitation_loss_on_batch(
                                        f_adaptation_policy,
                                        sub_batch,
                                        train_task_idx,
                                        inner=True,
                                    )
                                else:
                                    (
                                        loss,
                                        adv,
                                        weights,
                                        adv_loss,
                                    ) = self.adaptation_policy_loss_on_batch(
                                        f_adaptation_policy,
                                        adapted_q_function,
                                        adapted_value_function,
                                        sub_batch,
                                        train_task_idx,
                                        inner=True,
                                    )
                                    if adv_loss is not None:
                                        adv_policy_losses.append(adv_loss.item())
                                    inner_advantages.append(adv.item())
                                    inner_weights.append(weights.mean().item())

                                diff_policy_opt.step(loss)
                                inner_policy_losses.append(loss.item())

                        if self._args.imitation:
                            meta_policy_loss = self.imitation_loss_on_batch(
                                f_adaptation_policy, policy_meta_batch, train_task_idx
                            )
                        else:
                            (
                                meta_policy_loss,
                                outer_adv,
                                outer_weights_,
                                _,
                            ) = self.adaptation_policy_loss_on_batch(
                                f_adaptation_policy,
                                adapted_q_function,
                                adapted_value_function,
                                policy_meta_batch,
                                train_task_idx,
                            )
                            outer_weights.append(outer_weights_.mean().item())
                            outer_advantages.append(outer_adv.item())

                        (
                            meta_policy_loss / len(self.task_config.train_tasks)
                        ).backward()
                        meta_policy_losses.append(meta_policy_loss.item())
                        ##################################################################################################

                        # Sample adapted policy trajectory, add to replay buffer i [L12]
                        if train_step_idx % self._gradient_steps_per_iteration == 0:
                            (
                                adapted_trajectory,
                                adapted_reward,
                                success,
                            ) = self._rollout_policy(
                                f_adaptation_policy,
                                self._env,
                                sample_mode=self._args.offline,
                            )
                            train_rewards.append(adapted_reward)
                            successes.append(success)

                            if not (self._args.offline or self._args.offline_inner):
                                if self._args.sample_exploration_inner:
                                    exploration_trajectory, _, _ = self._rollout_policy(
                                        self._exploration_policy,
                                        self._env,
                                        sample_mode=False,
                                    )
                                    inner_buffer.add_trajectory(exploration_trajectory)
                                else:
                                    inner_buffer.add_trajectory(adapted_trajectory)
                            if not (self._args.offline or self._args.offline_outer):
                                outer_buffer.add_trajectory(adapted_trajectory)
                                # full_buffer.add_trajectory(adapted_trajectory)
                        else:
                            success = False

            if train_step_idx % self._gradient_steps_per_iteration == 0:
                if len(inner_value_losses):
                    if self._args.q:
                        writer.add_scalar(
                            f"Loss_Q_Inner/Task_{i}",
                            np.mean(inner_q_losses),
                            train_step_idx,
                        )
                    writer.add_scalar(
                        f"Loss_Value_Inner/Task_{train_task_idx}",
                        np.mean(inner_value_losses),
                        train_step_idx,
                    )
                    writer.add_scalar(
                        f"Loss_Policy_Inner/Task_{train_task_idx}",
                        np.mean(inner_policy_losses),
                        train_step_idx,
                    )
                    if len(adv_policy_losses):
                        writer.add_scalar(
                            f"Loss_Policy_Adv_Inner/Task_{train_task_idx}",
                            np.mean(adv_policy_losses),
                            train_step_idx,
                        )
                    writer.add_scalar(
                        f"Value_Mean_Inner/Task_{train_task_idx}",
                        np.mean(inner_values),
                        train_step_idx,
                    )
                    writer.add_scalar(
                        f"Advantage_Mean_Inner/Task_{train_task_idx}",
                        np.mean(inner_advantages),
                        train_step_idx,
                    )
                    writer.add_scalar(
                        f"Weight_Mean_Inner/Task_{train_task_idx}",
                        np.mean(inner_weights),
                        train_step_idx,
                    )
                    writer.add_scalar(
                        f"MC_Mean_Inner/Task_{train_task_idx}",
                        np.mean(inner_mc_means),
                        train_step_idx,
                    )
                    writer.add_scalar(
                        f"MC_std_Inner/Task_{train_task_idx}",
                        np.mean(inner_mc_stds),
                        train_step_idx,
                    )
                    writer.add_histogram(
                        f"Inner_Weights/Task_{train_task_idx}", weights, train_step_idx
                    )
                writer.add_scalar(
                    f"Value_Mean_Outer/Task_{train_task_idx}",
                    np.mean(outer_values),
                    train_step_idx,
                )
                writer.add_scalar(
                    f"Weight_Mean_Outer/Task_{train_task_idx}",
                    np.mean(outer_weights),
                    train_step_idx,
                )
                writer.add_scalar(
                    f"Advantage_Mean_Outer/Task_{train_task_idx}",
                    np.mean(outer_advantages),
                    train_step_idx,
                )
                writer.add_scalar(
                    f"MC_Mean_Outer/Task_{train_task_idx}",
                    np.mean(outer_mc_means),
                    train_step_idx,
                )
                writer.add_scalar(
                    f"MC_std_Outer/Task_{train_task_idx}",
                    np.mean(outer_mc_stds),
                    train_step_idx,
                )
                if self._args.q:
                    writer.add_scalar(
                        f"Loss_Q_Outer/Task_{train_task_idx}",
                        np.mean(meta_q_losses),
                        train_step_idx,
                    )
                writer.add_scalar(
                    f"Loss_Value_Outer/Task_{train_task_idx}",
                    np.mean(meta_value_losses),
                    train_step_idx,
                )
                writer.add_scalar(
                    f"Loss_Policy_Outer/Task_{train_task_idx}",
                    np.mean(meta_policy_losses),
                    train_step_idx,
                )
                writer.add_histogram(
                    f"Value_LRs",
                    F.softplus(torch.stack(self._value_lrs)),
                    train_step_idx,
                )
                writer.add_histogram(
                    f"Policy_LRs",
                    F.softplus(torch.stack(self._policy_lrs)),
                    train_step_idx,
                )
                if not self._args.imitation:
                    writer.add_histogram(
                        f"Outer_Weights/Task_{train_task_idx}",
                        outer_weights_,
                        train_step_idx,
                    )
                # if train_step_idx % self._visualization_interval == 0:
                #    writer.add_scalar(f'Reward_Test/Task_{train_task_idx}', test_reward, train_step_idx)
                writer.add_scalar(
                    f"Success_Train/Task_{train_task_idx}", int(success), train_step_idx
                )
                if train_step_idx % self._gradient_steps_per_iteration == 0:
                    writer.add_scalar(
                        f"Reward_Train/Task_{train_task_idx}",
                        adapted_reward,
                        train_step_idx,
                    )
                    writer.add_scalar(
                        f"Success_Train/Task_{train_task_idx}",
                        np.mean(success),
                        train_step_idx,
                    )

        if not self._args.imitation:
            if self._args.advantage_head_coef is not None:
                writer.add_scalar(
                    f"Adv_Coef", F.softplus(self._adv_coef).item(), train_step_idx
                )

            # Meta-update value function [L14]
            grad = self.update_model(
                self._value_function,
                self._value_function_optimizer,
                clip=self._grad_clip,
            )
            writer.add_scalar(f"Value_Outer_Grad", grad, train_step_idx)

            # Meta-update Q function [L14]
            if self._args.q:
                grad = self.update_model(
                    self._q_function, self._q_function_optimizer, clip=self._grad_clip
                )
                writer.add_scalar(f"Q_Outer_Grad", grad, train_step_idx)

        # Meta-update adaptation policy [L15]
        grad = self.update_model(
            self._adaptation_policy,
            self._adaptation_policy_optimizer,
            clip=self._grad_clip,
        )
        writer.add_scalar(f"Policy_Outer_Grad", grad, train_step_idx)

        if self._args.lrlr > 0:
            self.update_params(self._value_lrs, self._value_lr_optimizer)
            self.update_params(self._q_lrs, self._q_lr_optimizer)
            self.update_params(self._policy_lrs, self._policy_lr_optimizer)
            if self._args.advantage_head_coef is not None:
                self.update_params([self._adv_coef], self._adv_coef_optimizer)

        return (
            rollouts,
            test_rewards,
            train_rewards,
            meta_value_losses,
            meta_policy_losses,
            None,
            successes,
        )

    # @profile
    def train(self):
        log_path = f"{self._log_dir}/{self._name}"
        if os.path.exists(log_path):
            sep = "."
            existing = os.listdir(f"{self._log_dir}")
            idx = 0
            for directory in existing:
                if directory.startswith(self._name):
                    idx += 1
            print(f"Experiment output {log_path} already exists.")
            log_path = f"{self._log_dir}/{self._name}{sep}{idx}"
            self._name = f"{self._name}{sep}{idx}"

        print(f"Saving outputs to {log_path}")
        os.makedirs(log_path)

        with open(f"{log_path}/args.txt", "w") as args_file:
            json.dump(self._args.__dict__, args_file, indent=4, sort_keys=True)
        with open(f"{log_path}/tasks.pkl", "wb") as tasks_file:
            pickle.dump(self._env.tasks, tasks_file)
        tensorboard_log_path = f"{log_path}/tb"
        if not os.path.exists(tensorboard_log_path):
            os.makedirs(tensorboard_log_path)
        summary_writer = SummaryWriter(tensorboard_log_path)

        # Gather initial trajectory rollouts
        if not self._args.load_inner_buffer or not self._args.load_outer_buffer:
            behavior_policy = self._exploration_policy
            print("Using randomly initialized exploration policy")
            """
            if self._args.sampling_policy_path is not None:
                archive_path = self._args.sampling_policy_path
                if self._instance_idx > 0:
                    comps = archive_path.split('/')
                    comps[-2] += f'_{self._instance_idx}'
                    archive_path = '/'.join(comps)
                print(f'Loading behavior policy from path {archive_path}')
                behavior_policy = deepcopy(behavior_policy)
                behavior_policy.load_state_dict(torch.load(archive_path)['policy'])
            """
            if not self._args.online_ft:
                print("Gathering training task trajectories...")
                for i, (inner_buffer, outer_buffer) in enumerate(
                    zip(self._inner_buffers, self._outer_buffers)
                ):
                    while inner_buffer._stored_steps < self._args.initial_interacts:
                        task_idx = self.task_config.train_tasks[i]
                        self._env.set_task_idx(self.task_config.train_tasks[i])
                        if self._args.render_exploration:
                            print_(f"Task {task_idx}, trajectory {j}", self._silent)
                        trajectory, reward, success = self._rollout_policy(
                            behavior_policy, self._env, random=self._args.random
                        )
                        if self._args.render_exploration:
                            print_(f"Reward: {reward} {success}", self._silent)
                        if not self._args.load_inner_buffer:
                            inner_buffer.add_trajectory(trajectory, force=True)
                        if not self._args.load_outer_buffer:
                            outer_buffer.add_trajectory(trajectory, force=True)

            print("Gathering test task trajectories...")
            if not self._args.load_inner_buffer:
                for i, test_buffer in enumerate(self._test_buffers):
                    while test_buffer._stored_steps < self._args.initial_test_interacts:
                        task_idx = self.task_config.test_tasks[i]
                        self._env.set_task_idx(task_idx)
                        random_trajectory, _, _ = self._rollout_policy(
                            behavior_policy, self._env, random=self._args.random
                        )
                        test_buffer.add_trajectory(random_trajectory, force=True)

        if self._args.online_ft:
            print("Running ONLINE FINE-TUNING")
            self.eval_macaw(
                0,
                summary_writer,
                "online",
                ft_steps=500,
                steps_per_rollout=100,
                log_path=log_path,
            )
            print(f"Saved fine-tuning results to {tensorboard_log_path}")
            return

        rewards = []
        successes = []
        reward_count = 0
        for t in range(self._training_iterations):
            (
                rollouts,
                test_rewards,
                train_rewards,
                value,
                policy,
                vfs,
                success,
            ) = self.train_step(t, summary_writer)

            if not self._silent:
                if len(test_rewards):
                    # print_(f'{t}: {test_rewards}, {np.mean(value)}, {np.mean(policy)}, {time.time() - self._start_time}', self._silent)
                    print_("", self._silent)
                    print_(f"Step {t} Rewards:", self._silent)
                    for idx, r in enumerate(test_rewards):
                        print_(
                            f"Task {self.task_config.test_tasks[idx]}: {r}",
                            self._silent,
                        )
                    print_(f"MEAN TEST REWARD: {np.mean(test_rewards)}", self._silent)
                    print_(
                        f"Mean Value Function Outer Loss: {np.mean(value)}",
                        self._silent,
                    )
                    print_(f"Mean Policy Outer Loss: {np.mean(policy)}", self._silent)
                    print_(
                        f"Elapsed time (secs): {time.time() - self._start_time}",
                        self._silent,
                    )

                    if self._args.eval:
                        if reward_count == 0:
                            rewards = test_rewards
                            successes = [float(s) for s in success]
                        else:
                            factor = 1 / (reward_count + 1)
                            rewards = [
                                r + (r_ - r) * factor
                                for r, r_ in zip(rewards, test_rewards)
                            ]
                            print("*************")
                            print(success)
                            print("*************")
                            successes = [
                                s + (float(s_) - s) * factor
                                for s, s_ in zip(successes, success)
                            ]

                        reward_count += 1
                        print_(f"Rewards: {rewards}, {np.mean(rewards)}", self._silent)
                        print_(
                            f"Successes: {successes}, {np.mean(successes)}",
                            self._silent,
                        )
                        # if self._args.debug:
                        #    for idx, vf in enumerate(vfs):
                        #        print_(idx, argmax(vf, torch.zeros(self._observation_dim, device=self._device)), self._silent)

            if len(test_rewards):
                summary_writer.add_scalar(f"Reward_Test/Mean", np.mean(test_rewards), t)
            if len(train_rewards):
                summary_writer.add_scalar(
                    f"Reward_Train/Mean", np.mean(train_rewards), t
                )

                if self._args.target_reward is not None:
                    if np.mean(train_rewards) > self._args.target_reward:
                        print_("Target reward reached; breaking", self._silent)
                        break

            if t % self._visualization_interval == 0:
                try:
                    for idx, (env, rollout) in enumerate(zip(self._envs, rollouts)):
                        image = env.render_rollout(rollout, f"{log_path}/{t}_{idx}.png")
                except Exception as e:
                    pass

            if t % 1000 == 0:
                archive = {
                    "vf": self._value_function.state_dict(),
                    "vf_opt": self._value_function_optimizer.state_dict(),
                    "policy": self._adaptation_policy.state_dict(),
                    "policy_opt": self._adaptation_policy_optimizer.state_dict(),
                    "vf_lrs": self._value_lrs,
                    "q_lrs": self._q_lrs,
                    "policy_lrs": self._policy_lrs,
                }
                if self._args.advantage_head_coef is not None:
                    archive["adv_coef"] = self._adv_coef

                torch.save(archive, f"{log_path}/archive_LATEST.pt")
                if t % 10000 == 0:
                    torch.save(archive, f"{log_path}/archive_{t}.pt")

                if self._args.save_buffers:
                    for i, (inner_buffer, outer_buffer) in enumerate(
                        zip(self._inner_buffers, self._outer_buffers)
                    ):
                        print(f"{log_path}/outer_buffer_{i}.h5")
                        inner_buffer.save(f"{log_path}/inner_buffer_{i}.h5")
                        outer_buffer.save(f"{log_path}/outer_buffer_{i}.h5")
                        # full_buffer.save(f'{log_path}/full_buffer_{i}.h5')
