import argparse
import json
import pickle
import random
from collections import namedtuple

import numpy as np
import torch
from torch.multiprocessing import Process, set_start_method

from src.args import get_args
from src.envs import (AntDirEnv, HalfCheetahDirEnv, HalfCheetahVelEnv,
                      WalkerRandParamsWrappedEnv)
from src.macaw import MACAW


def run(args: argparse.Namespace, instance_idx: int = 0):
    with open(args.task_config, "r") as f:
        task_config = json.load(
            f, object_hook=lambda d: namedtuple("X", d.keys())(*d.values())
        )

    if args.advantage_head_coef == 0:
        args.advantage_head_coef = None

    tasks = []
    for task_idx in range(
        task_config.total_tasks if args.task_idx is None else [args.task_idx]
    ):
        with open(task_config.task_paths.format(task_idx), "rb") as f:
            task_info = pickle.load(f)
            assert len(task_info) == 1, f"Unexpected task info: {task_info}"
            tasks.append(task_info[0])

    seed = args.seed if args.seed is not None else instance_idx
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if task_config.env == "ant_dir":
        env = AntDirEnv(
            tasks, args.n_tasks, include_goal=args.include_goal or args.multitask
        )
    elif task_config.env == "cheetah_dir":
        env = HalfCheetahDirEnv(tasks, include_goal=args.include_goal or args.multitask)
    elif task_config.env == "cheetah_vel":
        env = HalfCheetahVelEnv(
            tasks,
            include_goal=args.include_goal or args.multitask,
            one_hot_goal=args.one_hot_goal or args.multitask,
        )
    elif task_config.env == "walker_params":
        env = WalkerRandParamsWrappedEnv(
            tasks, args.n_tasks, include_goal=args.include_goal or args.multitask
        )
    else:
        raise RuntimeError(f"Invalid env name {task_config.env}")

    if args.episode_length is not None:
        env._max_episode_steps = args.episode_length

    if args.name is None:
        args.name = "throwaway_test_run"
    if instance_idx == 0:
        name = args.name
    else:
        name = f"{args.name}_{instance_idx}"

    seed = args.seed if args.seed is not None else instance_idx
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model = MACAW(
        args,
        task_config,
        env,
        args.log_dir,
        name,
        training_iterations=args.train_steps,
        visualization_interval=args.vis_interval,
        silent=instance_idx > 0,
        instance_idx=instance_idx,
        gradient_steps_per_iteration=args.gradient_steps_per_iteration,
        replay_buffer_length=args.replay_buffer_size,
        discount_factor=args.discount_factor,
        seed=seed,
    )

    model.train()


if __name__ == "__main__":
    set_start_method("spawn")
    args = get_args()

    if args.instances == 1:
        if args.profile:
            import cProfile

            cProfile.runctx(
                "run(args)", sort="cumtime", locals=locals(), globals=globals()
            )
        else:
            run(args)
    else:
        for instance_idx in range(args.instances):
            subprocess = Process(target=run, args=(args, instance_idx))
            subprocess.start()
