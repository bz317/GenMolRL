#!/usr/bin/env python3
"""Plot QED distributions by reaction step for search baselines.

The plot uses one final/best molecule per test start. Greedy/random have one
saved final trajectory per start molecule. Exhaustive search can have many
paths and intermediate points per start, so it contributes only the best QED
available for that start across steps 1-5. Starts whose best option is taking no
action are shown in the step-0 column.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = ("Exhaustive", "GraphTransPPO", "PPO", "A2C", "TD3", "Greedy", "Random")
COLORS = {
    "Exhaustive": "#4C78A8",
    "Greedy": "#F58518",
    "Random": "#54A24B",
    "PPO": "#B279A2",
    "GraphTransPPO": "#72B7B2",
    "A2C": "#E45756",
    "TD3": "#9D755D",
}
# Short tick labels keep the per-step subplots thin enough for a 16:9 slide
# even with 7 methods.
DISPLAY_NAMES = {
    "Exhaustive": "Exhaustive",
    "Greedy": "Greedy",
    "Random": "Random",
    "PPO": "PPO",
    "GraphTransPPO": "GT-PPO",
    "A2C": "A2C",
    "TD3": "TD3",
}


def parse_exhaustive_best_action_steps(
    path: Path,
    max_step: int,
) -> tuple[dict[int, list[float]], dict[str, int]]:
    values = {step: [] for step in range(0, max_step + 1)}
    best_by_start: dict[str, tuple[int, float]] = {}
    start_qed_by_start: dict[str, float] = {}
    path_start_by_id: dict[int, str] = {}
    in_steps = False
    header: list[str] | None = None

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line == "[steps]":
                in_steps = True
                continue
            if not in_steps or not line:
                continue

            parts = line.split("\t")
            if header is None:
                header = parts
                continue
            if len(parts) < len(header):
                parts += [""] * (len(header) - len(parts))
            row = dict(zip(header, parts))

            try:
                path_id = int(row["path_id"])
                step = int(row["step"])
                qed = float(row["qed"])
            except (KeyError, ValueError):
                continue
            if step == 0:
                start_smiles = row["product"]
                path_start_by_id[path_id] = start_smiles
                start_qed_by_start.setdefault(start_smiles, qed)
                continue

            if not 1 <= step <= max_step:
                continue

            start_smiles = path_start_by_id.get(path_id)
            if start_smiles is None:
                continue
            current_best = best_by_start.get(start_smiles)
            if current_best is None or qed > current_best[1]:
                best_by_start[start_smiles] = (step, qed)

    no_action_best = 0
    for start_smiles, start_qed in start_qed_by_start.items():
        best = best_by_start.get(start_smiles)
        if best is None or start_qed >= best[1]:
            no_action_best += 1
            values[0].append(start_qed)
            continue
        step, qed = best
        values[step].append(qed)

    return values, {
        "starts_seen": len(start_qed_by_start),
        "no_action_best": no_action_best,
        "action_best": sum(len(values[step]) for step in values),
    }


def parse_trajectory_finals(path: Path, max_step: int) -> dict[int, list[float]]:
    values = {step: [] for step in range(0, max_step + 1)}
    in_trajectories = False
    header: list[str] | None = None

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line == "[trajectories]":
                in_trajectories = True
                continue
            if not in_trajectories or not line:
                continue

            parts = line.split("\t")
            if header is None:
                header = parts
                continue
            if len(parts) < len(header):
                parts += [""] * (len(header) - len(parts))
            row = dict(zip(header, parts))

            try:
                step = int(row["num_reactions"])
                qed = float(row["final_qed"])
            except (KeyError, ValueError):
                continue
            if 1 <= step <= max_step:
                values[step].append(qed)

    return values


def parse_cached_policy_eval(path: Path, max_step: int) -> dict[int, list[float]] | None:
    if not path.is_file():
        return None
    values = {step: [] for step in range(0, max_step + 1)}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                step = int(row["num_reactions"])
                qed = float(row["final_qed"])
            except (KeyError, ValueError):
                continue
            if 0 <= step <= max_step:
                values[step].append(qed)
    return values


def evaluate_sb3_policy(
    *,
    repo_root: Path,
    method: str,
    model_path: Path,
    config_path: Path,
    cache_path: Path,
    max_step: int,
) -> dict[int, list[float]]:
    cached = parse_cached_policy_eval(cache_path, max_step)
    if cached is not None:
        return cached

    sys.path.insert(0, str(repo_root))
    from genmolrl.algorithms.common import env_kwargs
    from genmolrl.config import load_config
    from genmolrl.envs.molecule_design_env import MoleculeDesignEnv

    config = load_config(config_path)
    config["algorithm"] = method
    env = MoleculeDesignEnv(**env_kwargs(config, eval_env=True))
    env.start_strategy.reset_cycle()
    n_eval = env.start_strategy.num_starts()

    if method == "PPO":
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(model_path, device="cpu")
    elif method == "A2C":
        from stable_baselines3 import A2C

        # Import registers the custom policy class needed to unpickle the model.
        import genmolrl.algorithms.a2c.policies  # noqa: F401

        model = A2C.load(model_path, device="cpu")
    else:
        raise ValueError(f"Unsupported SB3 method: {method}")

    values = {step: [] for step in range(0, max_step + 1)}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "episode", "start_smiles", "final_smiles", "initial_qed", "final_qed", "num_reactions"])
        for episode in range(n_eval):
            obs, info = env.reset()
            start_smiles = str(info["SMILES"])
            initial_qed = float(info["QED"])
            final_smiles = start_smiles
            final_qed = initial_qed
            num_reactions = 0
            done = False

            while not done:
                if method == "PPO":
                    action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
                else:
                    action, _ = model.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                if info.get("stop") or info.get("reaction_failed") or info.get("bad_template_index") is not None:
                    final_smiles = str(info.get("SMILES") or final_smiles)
                    final_qed = float(info.get("QED", final_qed))
                    break
                num_reactions = int(info.get("step", num_reactions))
                final_smiles = str(info["SMILES"])
                final_qed = float(info["QED"])

            if 0 <= num_reactions <= max_step:
                values[num_reactions].append(final_qed)
            writer.writerow([method, episode, start_smiles, final_smiles, initial_qed, final_qed, num_reactions])

    env.close()
    return values


def _infer_seq_hidden_dims(state_dict: dict, prefix: str) -> list[int] | None:
    """Recover ``hidden_dims`` from a saved ``nn.Sequential`` of Linear+Activation
    pairs followed by a final Linear output layer.

    Each Linear has key ``{prefix}.{i}.weight`` with shape ``(out, in)``; the
    hidden dims are the out-dims of every Linear *except* the last (output) one.
    Returns ``None`` if no matching layers are present (e.g. uni-discrete actor
    with no pi-net).
    """
    indices: list[int] = []
    for key in state_dict.keys():
        if not (key.startswith(prefix + ".") and key.endswith(".weight")):
            continue
        rest = key[len(prefix) + 1 : -len(".weight")]
        if "." in rest:
            continue
        try:
            indices.append(int(rest))
        except ValueError:
            continue
    if not indices:
        return None
    indices.sort()
    if len(indices) < 2:
        return []
    return [int(state_dict[f"{prefix}.{i}.weight"].shape[0]) for i in indices[:-1]]


def _infer_td3_arch(checkpoint: dict) -> dict:
    """Best-effort recovery of the actor/critic architecture from a TD3
    checkpoint produced by ``TD3Agent.save_model``.

    Activation isn't stored in the state dict, so we use the legacy actor
    width (``[256, 128, 128]``) as the heuristic for ReLU and the SB3-default
    ``[64, 64]`` shape as the heuristic for Tanh. This matches the two
    architectures any current checkpoint in this repo can have.
    """
    actor_sd = checkpoint.get("actor_state_dict", {})
    critic_sd = checkpoint.get("critic1_state_dict", {})
    actor_hidden = _infer_seq_hidden_dims(actor_sd, "f_net.network")
    pi_hidden = _infer_seq_hidden_dims(actor_sd, "pi_net.network")
    critic_hidden = _infer_seq_hidden_dims(critic_sd, "network")
    legacy_actor = [256, 128, 128]
    activation = "relu" if actor_hidden == legacy_actor else "tanh"
    return {
        "actor_hidden_dims": actor_hidden,
        "pi_hidden_dims": pi_hidden,
        "critic_hidden_dims": critic_hidden,
        "activation": activation,
    }


def evaluate_td3_policy(
    *,
    repo_root: Path,
    model_path: Path,
    config_path: Path,
    cache_path: Path,
    max_step: int,
) -> dict[int, list[float]]:
    """Roll out the trained TD3/PGFS agent on the eval start pool.

    Mirrors ``_evaluate_td3`` in ``genmolrl.algorithms.td3.train`` (deterministic
    actor argmax, KNN-wrapped env enabled) but writes per-episode trajectories
    to ``cache_path`` for the QED-by-step plot, instead of logging to wandb.
    """

    cached = parse_cached_policy_eval(cache_path, max_step)
    if cached is not None:
        return cached

    sys.path.insert(0, str(repo_root))
    import torch

    from genmolrl.algorithms.td3.agent import TD3Agent
    from genmolrl.algorithms.td3.mask_kind import td3_template_mask_kind
    from genmolrl.algorithms.td3.train import _make_td3_env
    from genmolrl.config import load_config

    config = load_config(config_path)
    config["algorithm"] = "TD3"
    env = _make_td3_env(config, eval_env=True)

    td3_cfg = config.get("td3", {})
    train_cfg = config.get("training", {})

    def _yaml_hidden_dims(key):
        value = td3_cfg.get(key)
        if value is None:
            return None
        return [int(x) for x in value]

    template_mask_cfg = td3_cfg.get("template_mask_kind")
    template_mask_kind = template_mask_cfg if isinstance(template_mask_cfg, str) else None

    target_entropy_ratio_cfg = td3_cfg.get("target_entropy_ratio")
    target_entropy_ratio = (
        None if target_entropy_ratio_cfg is None else float(target_entropy_ratio_cfg)
    )

    # The current YAML can disagree with what was on disk at training time
    # (the [64, 64] / Tanh keys were added on 2026-05-09; older checkpoints
    # were trained with the legacy [256, 128, 128] ReLU defaults). Recover the
    # actual training-time architecture from the saved state dict so the load
    # matches the parameter shapes either way.
    try:
        checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(model_path), map_location="cpu")
    inferred = _infer_td3_arch(checkpoint)

    yaml_actor = _yaml_hidden_dims("actor_hidden_dims")
    yaml_critic = _yaml_hidden_dims("critic_hidden_dims")
    yaml_pi = _yaml_hidden_dims("pi_hidden_dims")
    yaml_activation = td3_cfg.get("activation")
    if isinstance(yaml_activation, str):
        yaml_activation = yaml_activation.lower()

    actor_hidden = inferred["actor_hidden_dims"] if inferred["actor_hidden_dims"] is not None else yaml_actor
    pi_hidden = inferred["pi_hidden_dims"] if inferred["pi_hidden_dims"] else yaml_pi
    critic_hidden = inferred["critic_hidden_dims"] if inferred["critic_hidden_dims"] is not None else yaml_critic
    activation = inferred["activation"] if inferred["activation"] is not None else yaml_activation

    agent = TD3Agent(
        env,
        actor_lr=float(td3_cfg.get("actor_lr", 1e-4)),
        critic_lr=float(td3_cfg.get("critic_lr", 3e-4)),
        gamma=float(td3_cfg.get("gamma", 0.99)),
        tau=float(td3_cfg.get("tau", 0.005)),
        policy_noise=float(td3_cfg.get("policy_noise", 0.2)),
        noise_std=float(td3_cfg.get("noise_std", 0.1)),
        noise_clip=float(td3_cfg.get("noise_clip", 0.2)),
        policy_freq=int(td3_cfg.get("policy_freq", 2)),
        temperature_start=float(td3_cfg.get("initial_temperature", 1.0)),
        temperature_end=float(td3_cfg.get("min_temperature", 0.25)),
        start_timesteps=int(train_cfg.get("start_timesteps", 10000)),
        max_timesteps=int(train_cfg.get("total_timesteps", 1_000_000)),
        template_mask_kind=template_mask_kind,
        entropy_regularization=bool(td3_cfg.get("entropy_regularization", False)),
        entropy_alpha=float(td3_cfg.get("entropy_alpha", 0.2)),
        auto_tune_alpha=bool(td3_cfg.get("auto_tune_alpha", False)),
        target_entropy=float(td3_cfg.get("target_entropy", 0.5)),
        target_entropy_ratio=target_entropy_ratio,
        alpha_lr=float(td3_cfg.get("alpha_lr", 3e-4)),
        actor_hidden_dims=actor_hidden,
        critic_hidden_dims=critic_hidden,
        pi_hidden_dims=pi_hidden,
        activation=activation,
    )
    agent.apply_checkpoint(checkpoint, source_label=str(model_path))

    use_stop_action = bool(env.unwrapped.use_stop_action)
    num_templates = int(env.unwrapped.num_templates)

    def _has_real_action(smiles: str | None) -> bool:
        if not smiles:
            return False
        kind = td3_template_mask_kind(env, override=template_mask_kind)
        return bool(env.unwrapped.reaction_manager.feasible_first_reactant_templates(smiles, kind=kind))

    env.unwrapped.start_strategy.reset_cycle()
    n_eval = env.unwrapped.start_strategy.num_starts()

    values = {step: [] for step in range(0, max_step + 1)}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["method", "episode", "start_smiles", "final_smiles", "initial_qed", "final_qed", "num_reactions"]
        )
        for episode in range(n_eval):
            state, info = env.reset()
            start_smiles = str(info.get("SMILES", ""))
            initial_qed = float(info.get("QED", 0.0))
            final_smiles = start_smiles
            final_qed = initial_qed
            num_reactions = 0
            done = False

            while not done:
                if not use_stop_action and not _has_real_action(info.get("SMILES")):
                    break
                if hasattr(env, "enable"):
                    env.enable()
                action = agent.get_action(state, evaluate=True)
                template_idx = int(action[0].detach().reshape(-1).argmax().item())
                selected_stop = use_stop_action and template_idx == num_templates
                state, _reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                if (
                    selected_stop
                    or info.get("stop")
                    or info.get("reaction_failed")
                    or info.get("bad_template_index") is not None
                ):
                    final_smiles = str(info.get("SMILES") or final_smiles)
                    final_qed = float(info.get("QED", final_qed))
                    break
                num_reactions = int(info.get("step", num_reactions))
                final_smiles = str(info.get("SMILES") or final_smiles)
                final_qed = float(info.get("QED", final_qed))

            if 0 <= num_reactions <= max_step:
                values[num_reactions].append(final_qed)
            writer.writerow(
                ["TD3", episode, start_smiles, final_smiles, initial_qed, final_qed, num_reactions]
            )

    env.close()
    return values


def evaluate_graphtransppo_policy(
    *,
    repo_root: Path,
    model_path: Path,
    config_path: Path,
    cache_path: Path,
    max_step: int,
) -> dict[int, list[float]]:
    """Roll out the trained GraphTransPPO agent on the eval start pool.

    Mirrors :meth:`GraphTransPPO._greedy_trajectory` (deterministic argmax over
    masked logits, Stop / invalid-reaction termination) but records the
    absolute ``final_qed`` plus ``num_reactions`` per episode and writes a
    per-episode CSV cache for the QED-by-step plot.
    """
    cached = parse_cached_policy_eval(cache_path, max_step)
    if cached is not None:
        return cached

    sys.path.insert(0, str(repo_root))
    import torch

    from genmolrl.algorithms.graphtransppo.train import GraphTransPPO, _qed
    from genmolrl.config import load_config

    config = load_config(config_path)
    # Force CPU so the script also works on login nodes without a GPU; the
    # eval is small (one greedy trajectory per test SMILES) so the CPU path
    # is plenty fast and avoids a CUDA dependency for plot regeneration.
    config.setdefault("graphtransppo", {})["device"] = "cpu"
    trainer = GraphTransPPO(config)

    try:
        checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(model_path), map_location="cpu")
    state = checkpoint.get("policy", checkpoint)
    trainer.policy.load_state_dict(state)
    trainer.policy.eval()

    values = {step: [] for step in range(0, max_step + 1)}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "episode",
                "start_smiles",
                "final_smiles",
                "initial_qed",
                "final_qed",
                "num_reactions",
            ]
        )
        for episode, start_smiles in enumerate(trainer.sampler.eval_starts()):
            current = str(start_smiles)
            initial_qed = _qed(current, round_digits=trainer.qed_round_digits)
            num_reactions = 0
            with torch.no_grad():
                for _ in range(trainer.max_episode_len + int(trainer.use_stop_action)):
                    at_max = num_reactions >= trainer.max_episode_len
                    if at_max and not trainer.use_stop_action:
                        break
                    masked_logits, _, mask = trainer._forward_single(
                        current, force_stop=at_max
                    )
                    if not bool(mask.any()):
                        break
                    action = int(torch.argmax(masked_logits).item())
                    if action == trainer.stop_index:
                        break
                    product = trainer.reaction_manager.apply_reaction(
                        current, trainer.reaction_manager.templates[action], None
                    )
                    if product is None:
                        break
                    current = product
                    num_reactions += 1
            final_qed = _qed(current, round_digits=trainer.qed_round_digits)

            if 0 <= num_reactions <= max_step:
                values[num_reactions].append(final_qed)
            writer.writerow(
                [
                    "GraphTransPPO",
                    episode,
                    start_smiles,
                    current,
                    initial_qed,
                    final_qed,
                    num_reactions,
                ]
            )
    return values


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def write_summary_csv(
    output_path: Path,
    grouped_values: dict[str, dict[int, list[float]]],
    max_step: int,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "step", "n", "mean", "median", "q1", "q3", "min", "max"])
        for method in METHODS:
            for step in range(0, max_step + 1):
                values = grouped_values[method][step]
                if values:
                    writer.writerow(
                        [
                            method,
                            step,
                            len(values),
                            sum(values) / len(values),
                            median(values),
                            quantile(values, 0.25),
                            quantile(values, 0.75),
                            min(values),
                            max(values),
                        ]
                    )
                else:
                    writer.writerow([method, step, 0, "", "", "", "", "", ""])


def plot_boxplots(
    output_path: Path,
    grouped_values: dict[str, dict[int, list[float]]],
    max_step: int,
) -> None:
    # 16:9 figure tuned for full-bleed slides at 1920x1080 (16x9 in @120 dpi).
    # Each step subplot is therefore ~2.66" wide for max_step=5, which is
    # enough for the 7 method boxes once we shrink them below.
    fig, axes = plt.subplots(
        1,
        max_step + 1,
        figsize=(16, 9),
        sharey=True,
        constrained_layout=True,
    )
    if max_step == 0:
        axes = [axes]

    n_methods = len(METHODS)
    # Narrower boxes with a touch more horizontal spacing so all 7 methods
    # remain visually separated even though each step subplot is thinner.
    spacing = 0.11
    box_width = 0.07
    half_span = (n_methods - 1) * spacing / 2.0
    all_positions = [1.0 + (i - (n_methods - 1) / 2.0) * spacing for i in range(n_methods)]
    position_by_method = dict(zip(METHODS, all_positions))
    xlim = (1.0 - half_span - spacing, 1.0 + half_span + spacing)

    for axis_index, axis in enumerate(axes):
        step = axis_index
        present = [
            (method, grouped_values[method][step])
            for method in METHODS
            if grouped_values[method][step]
        ]
        data = [values for _, values in present]
        positions = [position_by_method[method] for method, _ in present]
        box = axis.boxplot(
            data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.2},
            boxprops={"linewidth": 0.9},
            whiskerprops={"linewidth": 0.9},
            capprops={"linewidth": 0.9},
        )
        for patch, (method, _) in zip(box["boxes"], present):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.72)

        if step == 0:
            axis.set_title("No action", fontsize=12)
        else:
            axis.set_title(
                f"After {step} reaction{'s' if step > 1 else ''}", fontsize=12
            )
        axis.set_xlabel("Method", fontsize=10)
        axis.set_xlim(*xlim)
        axis.set_xticks(all_positions)
        axis.set_xticklabels(
            [DISPLAY_NAMES[m] for m in METHODS], fontsize=8, rotation=45, ha="right"
        )
        axis.tick_params(axis="y", labelsize=9)
        axis.grid(axis="y", alpha=0.25)
        for method in METHODS:
            idx = position_by_method[method]
            values = grouped_values[method][step]
            # Vertical orientation keeps the per-method n labels readable when
            # 7 methods sit inside a thin 16:9 subplot.
            axis.text(
                idx,
                0.015,
                f"n={len(values)}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=7,
                transform=axis.get_xaxis_transform(),
            )

    axes[0].set_ylabel("QED", fontsize=11)
    fig.suptitle(
        "Final-molecule QED distributions by reaction count",
        fontsize=16,
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--exhaustive",
        type=Path,
        default=repo_root / "runs/exhausted_search_uni_results_5_len.txt",
    )
    parser.add_argument(
        "--greedy",
        type=Path,
        default=repo_root / "runs/greedy_search_uni_results.txt",
    )
    parser.add_argument(
        "--random",
        type=Path,
        default=repo_root / "runs/random_search_uni_results.txt",
    )
    parser.add_argument(
        "--ppo-model",
        type=Path,
        default=repo_root / "runs/06mjmn3t/wandb_model/model.zip",
    )
    parser.add_argument(
        "--a2c-model",
        type=Path,
        default=repo_root / "runs/lw78laao/wandb_model/model.zip",
    )
    parser.add_argument(
        "--td3-model",
        type=Path,
        default=repo_root / "runs/np18l9uf/td3_checkpoints/final_model.pth",
    )
    parser.add_argument(
        "--graphtransppo-model",
        type=Path,
        default=repo_root / "runs/v855kfxl/best_model.pt",
    )
    parser.add_argument(
        "--ppo-config",
        type=Path,
        default=repo_root / "configs/ppo_uni_masked_delta_qed.yaml",
    )
    parser.add_argument(
        "--a2c-config",
        type=Path,
        default=repo_root / "configs/a2c_uni_masked_delta_qed.yaml",
    )
    parser.add_argument(
        "--td3-config",
        type=Path,
        default=repo_root / "configs/td3_uni_continuous_masked_delta_qed.yaml",
    )
    parser.add_argument(
        "--graphtransppo-config",
        type=Path,
        default=repo_root / "configs/graphtransppo_uni_delta_qed.yaml",
    )
    parser.add_argument(
        "--ppo-cache",
        type=Path,
        default=repo_root / "visualization/ppo_06mjmn3t_eval_trajectories.csv",
    )
    parser.add_argument(
        "--a2c-cache",
        type=Path,
        default=repo_root / "visualization/a2c_lw78laao_eval_trajectories.csv",
    )
    parser.add_argument(
        "--td3-cache",
        type=Path,
        default=repo_root / "visualization/td3_np18l9uf_eval_trajectories.csv",
    )
    parser.add_argument(
        "--graphtransppo-cache",
        type=Path,
        default=repo_root / "visualization/graphtransppo_v855kfxl_eval_trajectories.csv",
    )
    parser.add_argument("--max-step", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "visualization/search_qed_by_step_boxplots.png",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=repo_root / "visualization/search_qed_by_step_summary.csv",
    )
    parser.add_argument(
        "--exhaustive-stats",
        type=Path,
        default=repo_root / "visualization/search_qed_exhaustive_best_action_stats.txt",
    )
    args = parser.parse_args()

    if args.max_step < 1 or args.max_step > 5:
        raise ValueError("--max-step must be between 1 and 5")

    exhaustive_values, exhaustive_stats = parse_exhaustive_best_action_steps(
        args.exhaustive,
        args.max_step,
    )
    grouped_values = {
        "Exhaustive": exhaustive_values,
        "Greedy": parse_trajectory_finals(args.greedy, args.max_step),
        "Random": parse_trajectory_finals(args.random, args.max_step),
        "PPO": evaluate_sb3_policy(
            repo_root=repo_root,
            method="PPO",
            model_path=args.ppo_model,
            config_path=args.ppo_config,
            cache_path=args.ppo_cache,
            max_step=args.max_step,
        ),
        "GraphTransPPO": evaluate_graphtransppo_policy(
            repo_root=repo_root,
            model_path=args.graphtransppo_model,
            config_path=args.graphtransppo_config,
            cache_path=args.graphtransppo_cache,
            max_step=args.max_step,
        ),
        "A2C": evaluate_sb3_policy(
            repo_root=repo_root,
            method="A2C",
            model_path=args.a2c_model,
            config_path=args.a2c_config,
            cache_path=args.a2c_cache,
            max_step=args.max_step,
        ),
        "TD3": evaluate_td3_policy(
            repo_root=repo_root,
            model_path=args.td3_model,
            config_path=args.td3_config,
            cache_path=args.td3_cache,
            max_step=args.max_step,
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    args.exhaustive_stats.parent.mkdir(parents=True, exist_ok=True)
    plot_boxplots(args.output, grouped_values, args.max_step)
    write_summary_csv(args.summary_csv, grouped_values, args.max_step)
    args.exhaustive_stats.write_text(
        "\n".join(f"{key}: {value}" for key, value in exhaustive_stats.items()) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.exhaustive_stats}")
    print(
        "Exhaustive starts best by taking no action: "
        f"{exhaustive_stats['no_action_best']} / {exhaustive_stats['starts_seen']}"
    )


if __name__ == "__main__":
    main()
