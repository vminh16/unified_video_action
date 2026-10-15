import gc
import sys

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
import numpy as np
import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
import random
import time
from omegaconf import OmegaConf, open_dict
from unified_video_action.workspace.base_workspace import BaseWorkspace
from unified_video_action.utils.load_env import load_env_runner


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
@click.option("--libero_task", default=None, help="LIBERO: evaluate only task files whose name contains this string")
@click.option("--debug", is_flag=True, help="LIBERO: evaluate only the first task file")
@click.option("-s", "--seed", type=int, default=None, help="torch/numpy seed of the policy sampling; default cfg.training.seed")
# env_runner overrides; None keeps the value stored in the checkpoint
@click.option("--n_test", type=int, default=None)
@click.option("--n_test_vis", type=int, default=None)
@click.option("--n_train", type=int, default=None)
@click.option("--n_train_vis", type=int, default=None)
@click.option("--n_envs", type=int, default=None, help="parallel envs; seeds are run in chunks of this size")
@click.option("--test_start_seed", type=int, default=None)
@click.option("--fix_goal", type=bool, default=None)
@click.option("--mp_context", type=click.Choice(["fork", "spawn", "forkserver"]), default=None, help="LIBERO: start method of the simulator workers")
@click.option("--seeded_goal", type=bool, default=None, help="PushT-M: goal drawn from the env seed (upstream: global np.random)")
@click.option("--act_diff_testing_steps", default=None, help="denoising steps of the action head at test time (checkpoint value if unset)")
def main(checkpoint, output_dir, device, libero_task, debug, seed, act_diff_testing_steps, **runner_overrides):

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load checkpoint
    # load on the CPU: CUDA must not be initialized before the env workers are forked (EGL rendering in LIBERO)
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]

    # set seed
    if seed is None:
        seed = cfg.training.seed
    print("seed:", seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    with open_dict(cfg):
        cfg.output_dir = output_dir
        # the workspace constructor reseeds from cfg.training.seed
        cfg.training.seed = seed
        if debug:
            cfg.training.debug = True
        if libero_task:
            cfg.task.eval_task_filter = libero_task
        if act_diff_testing_steps is not None:
            cfg.model.policy.autoregressive_model_params.act_diff_testing_steps = str(act_diff_testing_steps)
        for key, value in runner_overrides.items():
            if value is not None:
                cfg.task.env_runner[key] = value
    print("env_runner:", OmegaConf.to_container(cfg.task.env_runner))
    OmegaConf.save(cfg.task.env_runner, os.path.join(output_dir, "env_runner.yaml"))

    # configure workspace
    cls = hydra.utils.get_class(cfg.model._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace

    print("Loaded checkpoint from %s" % checkpoint)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    # evaluation only uses the EMA policy: free the payload, the raw model and the optimizer state
    # (they stay in host RAM since the checkpoint is loaded on the CPU)
    del payload
    workspace.model = None
    workspace.optimizer = None
    gc.collect()
    
    
    # create the env runners (they fork their workers) before the policy goes to the GPU
    print("cuda initialized before env runners:", torch.cuda.is_initialized())
    env_runners = load_env_runner(cfg, output_dir)

    # get policy from workspace
    policy = workspace.ema_model
    policy.to(device)
    policy.eval()

    # wall time of every predict_action call (one action chunk for every parallel env), CUDA-synchronized
    call_seconds, call_batch = [], []
    predict_action = policy.predict_action

    def timed_predict_action(obs_dict, *args, **kwargs):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        result = predict_action(obs_dict, *args, **kwargs)
        torch.cuda.synchronize(device)
        call_seconds.append(time.perf_counter() - t0)
        call_batch.append(int(next(iter(obs_dict.values())).shape[0]))
        return result

    policy.predict_action = timed_predict_action
    torch.cuda.reset_peak_memory_stats(device)

    if "libero" in cfg.task.name:
        step_log = {}
        for env_runner in env_runners:
            runner_log = env_runner.run(policy)
            step_log.update(runner_log)
            print(step_log)

        assert "test_mean_score" not in step_log
        all_test_mean_score = {
            k: v for k, v in step_log.items() if "test/" in k and "_mean_score" in k
        }
        step_log["test_mean_score"] = np.mean(list(all_test_mean_score.values()))

        runner_log = step_log
    else:
        env_runner = env_runners
        runner_log = env_runner.run(policy)

    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value

    for k, v in json_log.items():
        print(k, v)

    json.dump(dict(
        act_diff_testing_steps=cfg.model.policy.autoregressive_model_params.get("act_diff_testing_steps"),
        call_seconds=call_seconds,
        call_batch=call_batch,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3,
    ), open(os.path.join(output_dir, "latency.json"), "w"))

    out_path = os.path.join(output_dir, f'eval_log_{checkpoint.split("/")[-1]}.json')
    print("Saving log to %s" % out_path)
    json.dump(json_log, open(out_path, "w"), indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
