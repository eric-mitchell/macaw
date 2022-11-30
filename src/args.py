import argparse
import json


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling_policy_path", type=str, default=None)
    parser.add_argument("--online_ft", action="store_true")
    parser.add_argument("--imitation", action="store_true")
    parser.add_argument("--td3ctx", action="store_true")
    parser.add_argument("--buffer_mode", type=str, default="end")
    parser.add_argument("--value_reg", type=float, default=0)
    parser.add_argument("--contiguous", action="store_true")
    parser.add_argument("--archive", type=str, default=None)
    parser.add_argument("--wlinear", action="store_true")
    parser.add_argument("--macaw_params", type=str, default=None)
    parser.add_argument("--macaw_override_params", type=str, default=None)
    parser.add_argument("--target_vf_alpha", type=float, default=0.9)
    parser.add_argument("--bootstrap_grad", action="store_true")
    parser.add_argument("--buffer_skip", type=int, default=1)
    parser.add_argument("--inner_buffer_skip", type=int, default=1)
    parser.add_argument("--goal_dim", type=int, default=0)
    parser.add_argument("--info_dim", type=int, default=0)
    parser.add_argument("--multitask", action="store_true")
    parser.add_argument("--mt_value_lr", type=float, default=1e-2)
    parser.add_argument("--mt_policy_lr", type=float, default=1e-3)
    parser.add_argument("--pad_buffers", action="store_true")
    parser.add_argument("--task_batch_size", type=int, default=None)
    parser.add_argument("--action_sigma", type=float, default=0.2)
    parser.add_argument("--log_targets", action="store_true")
    parser.add_argument(
        "--traj_hold_out_test", dest="traj_hold_out_train", action="store_false"
    )
    parser.add_argument("--traj_hold_out_train", action="store_true", default=None)
    parser.add_argument("--trim_obs", type=int, default=None)
    parser.add_argument("--n_tasks", type=int, default=None)
    parser.add_argument("--multitask_eval", action="store_true")
    parser.add_argument("--multitask_bias_only", action="store_true")
    parser.add_argument("--lrlr", type=float, default=1e-3)
    parser.add_argument("--huber", action="store_true")
    parser.add_argument("--net_width", type=int, default=300)
    parser.add_argument("--net_depth", type=int, default=3)
    parser.add_argument("--cvae_prior_conditional", action="store_true")
    parser.add_argument("--cvae_preprocess", action="store_true")
    parser.add_argument("--trim_episodes", type=int, default=0)
    parser.add_argument("--episode_length", type=int, default=None)
    parser.add_argument("--normalize_values_outer", action="store_true")
    parser.add_argument("--normalize_values", action="store_true")
    parser.add_argument("--no_norm", action="store_true")
    parser.add_argument("--no_bootstrap", action="store_true")
    parser.add_argument("--q", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--render_exploration", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--train_exploration", action="store_true")
    parser.add_argument("--sample_exploration_inner", action="store_true")
    parser.add_argument("--cvae", action="store_true")
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--train_steps", type=int, default=1000000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--inner_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--inner_policy_lr", type=float, default=0.01)
    parser.add_argument("--inner_value_lr", type=float, default=0.01)
    parser.add_argument("--outer_policy_lr", type=float, default=1e-3)
    parser.add_argument("--outer_value_lr", type=float, default=1e-3)
    parser.add_argument("--exploration_lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--vis_interval", type=int, default=250)
    parser.add_argument("--log_dir", type=str, default="log")
    parser.add_argument("--include_goal", action="store_true")
    parser.add_argument("--single_task", action="store_true")
    parser.add_argument("--one_hot_goal", action="store_true")
    parser.add_argument("--task_idx", type=int, default=None)
    parser.add_argument("--instances", type=int, default=1)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--name_suffix", type=str, default="")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--gradient_steps_per_iteration", type=int, default=50)
    parser.add_argument("--replay_buffer_size", type=int, default=20000)
    parser.add_argument("--inner_buffer_size", type=int, default=20000)
    parser.add_argument("--full_buffer_size", type=int, default=20000)
    parser.add_argument("--discount_factor", type=float, default=0.99)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--initial_interacts", type=int, default=20000)
    parser.add_argument("--initial_test_interacts", type=int, default=512)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--offline_outer", action="store_true")
    parser.add_argument("--offline_inner", action="store_true")
    parser.add_argument(
        "--grad_clip", type=float, default=1e9
    )  # Essentially no clip, but use this to measure the size of gradients
    parser.add_argument("--exp_advantage_clip", type=float, default=20.0)
    parser.add_argument("--eval_maml_steps", type=int, default=1)
    parser.add_argument("--maml_steps", type=int, default=1)
    parser.add_argument("--adaptation_temp", type=float, default=1)
    parser.add_argument("--no_bias_linear", action="store_true")
    parser.add_argument("--advantage_head_coef", type=float, default=None)
    parser.add_argument("--entropy_alpha_coef", type=float, default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--target_reward", type=float, default=None)
    parser.add_argument("--save_buffers", action="store_true")
    parser.add_argument("--task_config", type=str, default=None)
    parser.add_argument("--load_inner_buffer", action="store_true")
    parser.add_argument("--load_outer_buffer", action="store_true")
    args = parser.parse_args()

    if args.macaw_params is not None:
        with open(args.macaw_params, "r") as f:
            print(f"Loading params from {args.macaw_params}")
            params = json.load(f)

        for k, v in params.items():
            setattr(args, k, v)

    if args.macaw_override_params is not None:
        with open(args.macaw_override_params, "r") as f:
            print(f"Loading OVERRIDE params from {args.macaw_override_params}")
            params = json.load(f)

        for k, v in params.items():
            setattr(args, k, v)

    if args.name_suffix:
        args.name = f"{args.name}_{args.name_suffix}"

    return args
