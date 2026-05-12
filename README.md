# GenMolRL

GenMolRL is the unified molecule-generation project for the current PPO, A2C, TD3/PGFS, GraphTransRL, GraphTransPPO, and Bi-PPO experiments. It replaces the previous workflow where PPO/A2C and TD3 were launched from different branch directories with duplicated environment, masking, data staging, reward, and logging logic.

The first goal is behavior-compatible migration of:

- `exp_branch_run_PPO_mask_extendedObs.sh`
- `exp_branch_run_A2C_mask.sh`
- `run_td3.sh`
- Random Search, Greedy Search, and Exhausted Search baselines (both uni and bi reaction modes)
- GraphTransRL with a graph-transformer policy backbone

The second goal is to host new methods built on top of the shared
environment / masking / staging stack:

- **GraphTransPPO** — graph-transformer backbone with PPO (clipped surrogate
  + value baseline) instead of the GraphTransRL trajectory-balance loss.
- **Bi-PPO** — two trainers for Bi reaction mode. The SB3 `MaskablePPO`
  variant uses `MultiDiscrete([T, R2])` with independent
  `π(T | R1) · π(R2 | R1)` heads. The hand-rolled variant supports either
  the independent parameterisation above or the autoregressive
  `π(T | R1) · π(R2 | R1, T)` (template-conditional R2), selectable via
  `ppo_bi.policy_arch`.
- Future methods such as REINVENT scaffold decorator slot in via the same
  `genmolrl/methods` adapter pattern.

## Layout

```text
GenMolRL/                       # <project-root> — `project_root()` resolves here
  genmolrl/
    chem/          # RDKit reactions, fingerprints, product selection, dataset staging
    envs/          # unified molecule-design Gymnasium env, masks, rewards, starts
    algorithms/    # PPO, A2C, TD3/PGFS, GraphTransRL, GraphTransPPO, ppo_bi trainers
                   # plus random / greedy / exhausted search baselines
    logging/       # W&B metrics and callbacks
    methods/       # lazy method adapters
    scripts/       # unified CLI entry points
  configs/         # experiment YAML configs (paths inside are project-relative)
  data/            # staged Uni/ and Bi/ datasets + templates
  runs/            # W&B run directories, monitors, checkpoints, search results
  tests/           # smoke tests
  tools/           # one-off audits, smoke checks, plot scripts
  run_launcher/
    sh_files/      # `run_genmolrl_*.sh` wrappers (set PROJECT_ROOT, cd, run python)
    HPC_scripts/
      uni/         # SLURM job scripts for uni-reaction launchers
      bi/          # SLURM job scripts for bi-reaction launchers
```

All wrapper scripts under `run_launcher/sh_files/` and SLURM scripts under
`run_launcher/HPC_scripts/{uni,bi}/` anchor themselves to the project root
above (via `$(dirname "$0")/../..` / a `SLURM_SUBMIT_DIR` fallback chain),
so they are safe to invoke from any cwd. All paths inside these scripts
and inside `configs/*.yaml` are relative to the project root.

## Installation

GenMolRL currently uses the repository-level conda environment exported at:

```text
conda_env_RL_for_new_mol.yml
```

Create the environment from scratch:

```bash
cd <repo-root>
conda env create -f conda_env_RL_for_new_mol.yml
conda activate RL_for_new_mol
```

If the environment already exists and you want to update it from the file:

```bash
cd <repo-root>
conda env update -n RL_for_new_mol -f conda_env_RL_for_new_mol.yml --prune
conda activate RL_for_new_mol
```

The environment includes the main dependencies used by the current PPO/A2C/TD3/search experiments, including RDKit, PyTorch, Stable-Baselines3, sb3-contrib, W&B, and FAISS GPU packages. If you run on a machine without a compatible GPU/CUDA setup, TD3 can still import but FAISS/KNN or CUDA execution may fall back or need environment-specific adjustment.

GraphTransRL additionally needs `torch_geometric` compatible with the installed PyTorch/CUDA build, because its policy uses `GENConv` and `TransformerConv`. For the current `torch==2.3.0+cu121` environment, install it with:

```bash
python -m pip install torch-geometric -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

GenMolRL is not installed as a site package by default. Run the wrapper scripts from the project root — they set `PYTHONPATH` and `cd` to the project root automatically:

```bash
cd <project-root>
./run_launcher/sh_files/run_genmolrl_ppo.sh
```

Optional editable install:

```bash
cd <project-root>
python -m pip install -e .
```

After an editable install, the explicit `PYTHONPATH=.` is usually no longer necessary, but the wrapper scripts still set it explicitly for portability.

For W&B logging, either log in normally:

```bash
wandb login
```

or drop the API key in a plain-text file **outside** the project (so it never
ends up in the git history or in a pushed artifact). The wrapper scripts look
for it at `<repo-root>/wandb_api_key.txt` — i.e. the parent of `<project-root>`:

```text
<repo-root>/wandb_api_key.txt          # NOT inside <project-root> = <repo-root>/GenMolRL/
```

`wandb_api_key.txt` is also listed in `.gitignore`, but the canonical location
is one directory above the project so an accidental `git add .` cannot stage
it. Override the path per-invocation by exporting `WANDB_API_KEY_FILE=/path/...`,
or skip the file entirely by exporting `WANDB_API_KEY` directly. To disable
cloud logging for smoke tests:

```bash
WANDB_MODE=disabled ./run_launcher/sh_files/run_genmolrl_ppo.sh
```

Quick installation check:

```bash
cd <project-root>          # i.e. <repo-root>/GenMolRL
conda activate RL_for_new_mol
PYTHONPATH=. python -m compileall -q genmolrl
```

## Data

GenMolRL is self-contained when the following files are present:

```text
data/Uni/reactants_train.pkl
data/Uni/reactants_test.pkl
data/Uni/templates_unimolecolar_explicit.pkl
```

Check the staged files with (run from the project root, `<repo>/GenMolRL`):

```bash
PYTHONPATH=. python -m genmolrl.scripts.stage_data
```

The command above only validates and reports the existing `data/Uni` layout. It does not depend on the old source tree. If you intentionally want to regenerate the staged files from an external split, pass that source explicitly:

```bash
PYTHONPATH=. python -m genmolrl.scripts.stage_data --source-dir <external-uni-split-dir>
```

The canonical Uni directory, relative to the `GenMolRL/` project root, is:

```text
data/Uni/reactants_train.pkl
data/Uni/reactants_test.pkl
data/Uni/templates_unimolecolar_explicit.pkl
```

It also creates derived compatibility files in the same directory:

```text
data/Uni/reactants_full.pkl
data/Uni/eval_start_smiles.txt
```

`reactants_train.pkl` is used for training, and `reactants_test.pkl` is used for testing/search and full test-set evaluation. Evaluation cycles through every molecule in `reactants_test.pkl` directly. `reactants_full.pkl` and `eval_start_smiles.txt` are staged as compatibility files, but they are not used by the current default configs.

The future Bi dataset should live under the `GenMolRL/` project root at:

```text
data/Bi/
```

That directory is intentionally empty for now. The spelling `unimolecolar` is preserved because existing data and scripts use that filename.

## Running Experiments

Use the wrapper scripts from the project root (`<repo>/GenMolRL`):

```bash
./run_launcher/sh_files/run_genmolrl_ppo.sh                # uni PPO (MaskablePPO Discrete)
./run_launcher/sh_files/run_genmolrl_a2c.sh                # uni A2C
./run_launcher/sh_files/run_genmolrl_td3_continuous.sh     # uni TD3, PGFS continuous R2 head
./run_launcher/sh_files/run_genmolrl_td3_discrete.sh       # uni TD3, template-only critic
./run_launcher/sh_files/run_genmolrl_graphtransrl.sh             # uni GraphTransRL (trajectory balance)
./run_launcher/sh_files/run_genmolrl_graphtransppo.sh            # uni GraphTransPPO (PPO over the graph backbone)
./run_launcher/sh_files/run_genmolrl_ppo_bi_multidiscrete.sh     # bi PPO, SB3 MaskablePPO + MultiDiscrete([T+1, R2])
./run_launcher/sh_files/run_genmolrl_ppo_bi_hierarchical.sh      # bi PPO, hand-rolled trainer with π(T|R1)·π(R2|R1,T)
./run_launcher/sh_files/run_genmolrl_random_search.sh
./run_launcher/sh_files/run_genmolrl_greedy_search.sh
./run_launcher/sh_files/run_genmolrl_exhausted.sh          # uni exhaustive enumeration
./run_launcher/sh_files/run_genmolrl_exhausted_bi.sh       # bi exhaustive enumeration (per-path streaming)
```

Or call the unified launcher directly (the YAML config alone fully
specifies the run; CLI flags are only needed for one-off overrides):

```bash
PYTHONPATH=. python -m genmolrl.scripts.run_experiment \
  --algorithm ppo \
  --config configs/ppo_uni_masked_delta_qed.yaml
```

### YAML is the single source of truth

To change `reaction_mode`, `masking`, `reward`, `experiment_name`,
`max_episode_len`, the dataset paths, or any model hyperparameter for a run,
**edit the YAML config**. The launcher scripts under `run_launcher/sh_files/`
no longer pass `${VAR:-default}` defaults to `run_experiment.py`; they only
forward `--algorithm` and `--config`, plus any of these optional flags **when
the matching env var is explicitly set**:

| Env var            | Forwarded as           | Purpose (when set)                                          |
| ------------------ | ---------------------- | ----------------------------------------------------------- |
| `REACTION_MODE`    | `--reaction-mode`      | Override `reaction_mode` for this invocation only.          |
| `MASKING`          | `--masking`            | Override `masking` for this invocation only.                |
| `REWARD`           | `--reward`             | Override `reward` for this invocation only.                 |
| `EXPERIMENT_NAME`  | `--experiment-name`    | Override the W&B run name (otherwise comes from YAML).      |
| `MAX_EPISODE_LEN`  | `--max-episode-len`    | Override `max_episode_len` for this invocation only.        |
| `TRAINING_FILE`    | `--training-file`      | Override `dataset.training_file`.                           |
| `TEST_FILE`        | `--test-file`          | Override `dataset.test_file`.                               |
| `TEMPLATES_FILE`   | `--templates-file`     | Override `dataset.templates_file`.                          |
| `GREEDY_MODE`      | `--greedy-mode`        | Override `search.greedy_mode` for greedy_search only.       |
| `CONFIG`           | `--config`             | Use a non-default YAML file.                                |
| `WANDB_RUN_POST_APPEND` | (handled in Python)  | Append a suffix to the resolved run name (e.g. `_HPC` from SLURM). |

If none of these are set, the YAML config is the only thing controlling the
run. This means: **changing a YAML value is enough** — you do not need to edit
the launcher script or the SLURM script. Common ad-hoc overrides:

```bash
MASKING=none ./run_launcher/sh_files/run_genmolrl_ppo.sh
REWARD=final_qed ./run_launcher/sh_files/run_genmolrl_a2c.sh
REACTION_MODE=bi ./run_launcher/sh_files/run_genmolrl_td3_continuous.sh
EXPERIMENT_NAME=my_test ./run_launcher/sh_files/run_genmolrl_ppo.sh
CONFIG=configs/td3_uni_masked_balance_delta_qed.yaml ./run_launcher/sh_files/run_genmolrl_td3_continuous.sh
MAX_EPISODE_LEN=3 ./run_launcher/sh_files/run_genmolrl_ppo.sh
WANDB_MODE=disabled ./run_launcher/sh_files/run_genmolrl_random_search.sh
```

Search runners also accept dataset path overrides:

```bash
TEST_FILE=data/Uni/reactants_test.pkl \
TEMPLATE_FILE=data/Uni/templates_unimolecolar_explicit.pkl \
./run_launcher/sh_files/run_genmolrl_random_search.sh
```

`TEMPLATES_FILE` is also accepted. Search baselines are test-only, so they use `TEST_FILE` and do not require a train data path. The GraphTransRL wrapper accepts `TRAINING_FILE`, `TEST_FILE`, and `TEMPLATES_FILE`. The wrapper scripts use the files already under `data/Uni` by default. Set `STAGE_DATA=true` to validate the staged files before launch, or set both `STAGE_DATA=true` and `STAGE_SOURCE_DIR=<external-uni-split-dir>` when you intentionally want to regenerate the staged files from an external source directory.

## Supported Algorithms

### PPO

`ppo` trains a policy with Stable-Baselines3 / sb3-contrib. Masked mode uses `MaskablePPO`; no-mask mode uses plain SB3 PPO.

### A2C

`a2c` trains an actor-critic policy. Masked mode uses a custom policy that reads the action mask from the observation tail; no-mask mode uses plain `MlpPolicy`.

### TD3/PGFS

`td3` trains the custom PGFS-style TD3 implementation. It learns a template selector plus a continuous R2 vector for bimolecular reactions.

#### Uni `delta_qed` convergence ceiling — diagnosis & fix history

On the Uni `delta_qed` benchmark with the standard 12,689-molecule test set
(`mean_start_qed ≈ 0.693`), TD3 reproducibly plateaus at
`eval/mean_final_delta_qed ≈ -0.018` (Stop disabled) or collapses to
`eval/stop_rate = 1.0` (Stop enabled), while PPO/A2C reach roughly `+0.04`
on the same benchmark.

##### High-level diagnosis

- TD3 is a deterministic-greedy actor that selects `argmax_a Q(s, a)` at
  evaluation time. With `reward: delta_qed`, average reactions on
  already-drug-like starts are slightly negative, so:
    - with Stop available, `Q(Stop) ≈ 0` sits in the middle of the
      per-template Q distribution and the actor's argmax collapses to
      "always Stop" (`eval/stop_rate = 1`, `eval/mean_reward = 0`);
    - with Stop disabled, the actor must react even on starts where every
      valid template lowers QED, so `eval/mean_final_delta_qed` is
      anchored by the ~60 % of starts whose best template is still
      negative.
- PPO/A2C avoid both traps with a stochastic categorical policy plus a
  value baseline: their gradient amplifies above-baseline templates
  *relative to the value baseline*, and a per-state nuanced "Stop on
  high-QED, react on low-QED" policy emerges naturally.

##### Things tried under `delta_qed` (and why they did not break the ceiling on their own)

- ε-greedy template injection during training rollouts
  (`td3.training_random_action_prob` decaying 0.3 → 0.05 over 500k
  steps): increases buffer diversity but does not change the actor's
  argmax preferences.
- Larger Gumbel-Softmax temperature (`initial_temperature: 2.0`,
  `min_temperature: 0.7`): only affects training-time stochasticity,
  argmax at eval is unchanged.
- `start_timesteps` raised from 10k to 200k of purely-random warm-up:
  more replay coverage but the post-warmup actor still collapses within
  ~10k actor updates (run `pthux4sd` shows the random warm-up actor
  scoring `pos_frac = 0.249`, then dropping to `0.0002` after 10k actor
  updates).
- **Conditional Stop penalty** in the env: keep `use_stop_action: true`
  but apply `stop_early_penalty` only when feasible reactions existed at
  the time Stop was selected. The penalty was driven up to `-0.05`
  applied at every step `≤ 4` of the 5-step episode (run `pthux4sd`).
  Did not help on its own: the bootstrap loop drags `V(s')` down by
  roughly the same magnitude as `Q(Stop)`, so the actor's argmax still
  globally prefers Stop. Code path is preserved (see
  `rewards.stop_reward`) but disabled by default in the YAMLs.
- **Entropy regularization** of the actor (SAC-discrete style with
  `entropy_regularization: true`), with **auto-tuned α** and a
  **per-state target entropy ratio** (`target_entropy_ratio = 0.5`,
  i.e. `H_target(s) = 0.5 · log N_feasible(s)`). Auto-tune kept policy
  entropy to within ±0.01 of target throughout training, but the actor's
  *argmax* (the eval policy) still picked Stop on 99.97 % of states —
  template logits stayed below Stop globally (run `pthux4sd`).
- **`reward: final_qed`** as a structural workaround (run `2va9u7gz`):
  cumulative-QED rewards remove the "Stop is the easy default" trap by
  making `Q(template)` ~1–3 vs `Q(Stop) = 0`, but introduce the
  *opposite* failure mode — the agent always reacts even when every
  available template lowers QED, because every successful template step
  emits positive reward. `mean_final_delta_qed` got worse, not better
  (-0.06 vs random's -0.022 baseline). Reverted.

##### Implementation audit (May 2026)

A systematic audit of `td3/agent.py`, `td3/train.py`,
`td3/replay_buffer.py`, and `envs/molecule_design_env.py` did not find a
correctness bug, but did identify two **structural asymmetries** with
PPO/A2C that bias TD3 toward the always-Stop trap. Both are now fixed:

1. **TD3 observation now includes the action mask, matching PPO/A2C.**
   Previously `algorithms/td3/train.py` hard-coded
   `kwargs["append_action_mask_to_obs"] = False`, so the actor's `f_net`
   only saw a 1024-dim Morgan fingerprint and had to *infer* which
   templates were feasible at each state. The Stop slot is always
   feasible, so its logit received gradient signal from every state in
   the batch while each template logit only got signal from states where
   that template was feasible — biasing the shared actor head toward
   "Stop logit > all template logits" globally. The new flow appends the
   16-dim mask to the observation (1024 → 1040 dims), matching PPO/A2C
   exactly. The continuous R2 head is decoupled via
   ``env.unwrapped.base_obs_dim`` so the second-reactant fingerprint
   stays at 1024 dims (see `td3.train._td3_fp_dim`).
2. **Replay buffer shrunk from 500k to 100k.** With
   `start_timesteps=200k` warmup transitions and only one transition per
   env step thereafter, a 500k buffer remains >80 % warmup-dominated for
   most of training. 100k means the buffer fully transitions to
   post-warmup samples within ~100k post-warmup steps so the actor sees
   the consequences of its own behavior on the buffer composition more
   quickly.

PPO/A2C, GraphTransRL, and the search baselines are unaffected (the obs
change lives entirely in `td3.train._make_td3_env`; the buffer change
lives entirely in the TD3 YAMLs). Reward type is back to `delta_qed`.

##### Deeper alignment pass (May 2026, second iteration)

After the first audit fixes, continuous TD3 (W&B run `np18l9uf`) started
moving in the right direction (`eval/mean_final_delta_qed ≈ +0.010`,
`eval/mean_ep_length` growing 1.4 → 1.77, `positive_delta_fraction`
growing from 22 % → 32 %), while discrete TD3 (`op4q1oxe`) stayed flat at
`eval/stop_rate = 1.0`. A second deep comparison against PPO/A2C surfaced
four further misalignments, all now resolved in the TD3 configs and the
agent code:

1. **Training pool aligned to `reactants_train.pkl` (no leakage).** Earlier
   PPO / A2C / GraphTransRL configs all pointed `dataset.training_file`
   at `reactants_full.pkl`, which is the union of `reactants_train.pkl`
   (n = 50 756) and `reactants_test.pkl` (n = 12 689) — so every test
   starting molecule was also a possible training start. TD3 was on the
   clean `train.pkl` split. The fix is the opposite of what is naively
   suggested by "make TD3 match the others": switch *all* algorithms to
   train on `reactants_train.pkl` only, so the 12 689-molecule eval pool
   is genuinely held-out for PPO / A2C / GraphTransRL / TD3 alike. Five
   YAMLs were updated (`ppo_uni_masked_delta_qed.yaml`,
   `a2c_uni_masked_delta_qed.yaml`, `graphtransrl_uni_delta_qed.yaml`,
   `td3_uni_continuous_masked_delta_qed.yaml`,
   `td3_uni_discrete_masked_delta_qed.yaml`). Any TD3 / PPO / A2C
   numbers logged before this commit were trained on a leaky pool and
   should be re-run before quoting.
2. **`reward_round_digits = 3` and `info_qed_round_digits = 3`.** PPO/A2C
   already round ΔQED to 3 decimals before the policy ever sees it,
   removing 4th-decimal RDKit-numerical noise from the regression
   target. TD3 was previously fitting full-precision rewards, so the
   critic was learning a noticeably noisier target. Both TD3 YAMLs now
   use the same rounding.
3. **`actor.f_net` (current actor) used in the soft Bellman target.**
   The entropy-regularized branch in `td3/agent.py::_train_entropy` was
   computing `next_logits = self.actor_target.f_net(next_state_obs)`, a
   leftover from vanilla TD3's continuous-action target smoothing. The
   reference SAC-discrete formulation uses the *current* actor in
   V_soft (only the critic is target-Polyak-averaged). With τ = 0.005
   the target actor lagged the policy by ~200 gradient steps, producing
   a stale entropy bonus and — more importantly for our failure mode —
   a stale next-action distribution that kept reinforcing
   `Q(s', Stop)` even after the current actor had begun to put mass on
   a template. Switching to the current actor removes that bias and
   aligns the branch with Christodoulou (2019). The deterministic
   branch (`_train_deterministic`) is unchanged: it legitimately uses
   `actor_target` for TD3-style action smoothing.
4. **Network arch aligned to PPO/A2C (`[64, 64]` Tanh).** PPO/A2C use
   SB3's `MlpPolicy` defaults, which is `[64, 64]` Tanh for both policy
   and value heads. TD3 was using `[256, 128, 128]` ReLU for the actor
   and `[256, 64, 16]` ReLU for the critic — roughly 4–5× more
   parameters than PPO/A2C and an unbounded activation that lets Q
   estimates drift, which is harmful in our small-reward regime
   (typical |ΔQED| ≈ 0.01). The TD3 agent now reads `actor_hidden_dims`,
   `critic_hidden_dims`, `pi_hidden_dims`, and `activation` from the
   YAML; both TD3 configs default to `[64, 64]` Tanh for actor and
   critic. The continuous R2 head (`pi_net`) has no PPO/A2C analog and
   stays at the legacy `[256, 256, 167]` ReLU unless the user opts in.
   When all knobs are unset (as in any pre-existing test fixture or
   external YAML) the legacy widths/activation are restored bit-for-bit
   for backward compatibility (covered by
   `tests/test_env_smoke.py::test_td3_agent_arch_alignment_with_ppo`).

After (1)–(4), the discrete TD3 actor `f_net` collapses to ~72 k
parameters (from ~318 k) and the critic to ~72 k (from ~133 k),
matching PPO/A2C's policy/value head sizes within ~1 %. Continuous TD3
keeps the larger R2 head, so it is still ~620 k actor parameters
(`f_net=72 k + pi_net=550 k`).

These four fixes only change the TD3 path; PPO/A2C/GraphTransRL/search
configs, agents, and behavior are byte-identical to before.

### GraphTransRL

`graphtransrl` trains a GenMolRL-owned graph-transformer RL policy. The method does not learn the first reactant: training episodes start from random molecules sampled from `dataset.training_file`, and each eval pass cycles through every molecule in `dataset.test_file` exactly once.

The policy backbone is a graph transformer with PyG `GENConv` and `TransformerConv` layers. Action logits cover reaction templates plus Stop; there is no exposed `AddFirstReactant` action path in GenMolRL. Rewards are per-action `delta_qed`, and `eval/mean_reward` is the mean summed per-action delta-QED over all test starts. The eval logs also include `eval/avg_delta_qed`, `eval/mean_final_delta_qed`, `eval/max_qed`, `eval/mean_ep_length`, and `eval/n_molecules`.

Run it with:

```bash
./run_launcher/sh_files/run_genmolrl_graphtransrl.sh
```

### GraphTransPPO

`graphtransppo` swaps the GraphTransRL trajectory-balance loss for a hand-
rolled PPO objective (clipped surrogate, GAE, value baseline, target-KL
early stop) over the *same* graph-transformer backbone (`GENConv` +
`TransformerConv`). A small value head is attached to the trunk so the
critic and policy share the encoder; the action head is the original
graph-attention readout over templates + Stop. Default config is
`configs/graphtransppo_uni_delta_qed.yaml` with `masking: reaction_valid`,
`reward: delta_qed`, and PPO knobs tuned to match `ppo_uni_masked_delta_qed.yaml`
so any GraphTransPPO ↔ PPO_Uni delta is attributable to the encoder rather
than the learning rule.

Run it with:

```bash
./run_launcher/sh_files/run_genmolrl_graphtransppo.sh
```

Currently uni-only. Bi support would require extending the graph readout to
produce a `(T, R2)` joint distribution; not implemented.

### PPO Bi-reaction

Two trainers / launchers exist for Bi reaction mode:

1. **`run_genmolrl_ppo_bi_multidiscrete.sh`** →
   `--algorithm ppo` + config `configs/ppo_bi_multidiscrete_delta_qed.yaml`.
   Uses SB3 `MaskablePPO` with action space
   `MultiDiscrete([num_templates + 1, num_reactants])`. Template and R2
   are sampled independently. The action mask is provided through
   `env.action_masks()` (not appended to the observation, to keep the
   ~1024-dim Morgan FP small).

2. **`run_genmolrl_ppo_bi_hierarchical.sh`** →
   `--algorithm ppo_bi` + config `configs/ppo_bi_hierarchical_delta_qed.yaml`.
   Hand-rolled PPO trainer that supports two policy architectures via the
   `ppo_bi.policy_arch` field. In both architectures the only state input
   is the Morgan fingerprint of the current molecule — i.e. `R1` for the
   next bi-reaction step. Equivalently, `s := fp(R1)` and `π(... | s)` and
   `π(... | R1)` refer to the same distribution.
   - `hierarchical` (default):
     `π(T, R2 | R1) = π_T(T | R1) · π_R2(R2 | R1, T)`.
     The R2 query MLP consumes the concatenation of the shared trunk
     features (encoding `R1`) and a learned per-template embedding of `T`,
     then dot-products against per-reactant embeddings — so R2 logits are
     a function of **both R1 and T**. The R2 mask is per-`(R1, T)`.
   - `multidiscrete`: matches the SB3 parameterisation but lives in the
     same hand-rolled trainer:
     `π(T, R2 | R1) = π_T(T | R1) · π_R2(R2 | R1)`. The R2 query depends
     on `R1` only; `T` is dropped from the R2 head. The R2 mask is the
     per-`R1` union over valid templates. Useful as a within-trainer
     ablation against the autoregressive default.

Masking semantics follow the README contract (see *Masking Modes*
below). Both launchers default to `masking=r2_available` for
wallclock-cheap training. Set `MASKING=reaction_valid` to enable the
zero-`invalid_reaction_penalty` contract:

- The multidiscrete launcher under `reaction_valid` benefits from the
  bi-aware `template_reaction_valid_mask` (`R1 match + ∃R2 in pool`) on
  the template axis, but `MaskablePPO` does not enforce joint
  `(T, R2)` validity — independent sampling can still emit `-1`.
- The hierarchical launcher under `reaction_valid` uses
  `ReactionManager.bi_r2_valid_mask` for the per-(state, T) R2 mask, so
  every sampled `(T, R2)` produces a sanitised product — zero `-1` by
  mask construction. The hand-rolled trainer's `multidiscrete` arch
  adds an online joint-rejection loop to preserve the same contract.

`substructure` and `r2_available` are pattern-only and explicitly allow
rare `apply_reaction` failures to surface as `-1` transitions for the
policy to learn from.

Run them with:

```bash
./run_launcher/sh_files/run_genmolrl_ppo_bi_multidiscrete.sh                          # SB3 MaskablePPO MultiDiscrete, r2_available
./run_launcher/sh_files/run_genmolrl_ppo_bi_hierarchical.sh                           # hand-rolled BiPPO Hierarchical, r2_available
MASKING=reaction_valid ./run_launcher/sh_files/run_genmolrl_ppo_bi_hierarchical.sh    # zero -1 contract
```

Corresponding SLURM scripts live in `run_launcher/HPC_scripts/bi/`:

```bash
sbatch run_launcher/HPC_scripts/bi/slurm_genmolrl_gpu_ppo_multidiscrete
sbatch run_launcher/HPC_scripts/bi/slurm_genmolrl_gpu_ppo_hierarchical
```

### Random Search

`random_search` is a non-neural baseline. It goes through `dataset.test_file` one molecule at a time, matching evaluation order. For each test start molecule, at each step:

1. Build the valid template list using the configured masking mode.
2. Randomly choose one valid template.
3. If the template is bimolecular, randomly choose one valid R2 from the reactant pool.
4. Apply the reaction and continue from the product if valid.
5. Stop when no valid action is available or `max_episode_len` is reached.

It writes a `.txt` report with `[summary]`, `[trajectories]`, and `[steps]` sections. The summary includes `max_qed` over all test starts and generated products, plus `avg_delta_qed`, the average of `final_qed - start_qed` over all test start molecules.

The text report includes a `START` row for every test molecule (`step=0`) followed by one row per successful reaction. If a start molecule has no valid action, it is still saved as a start-only trajectory with `delta_qed = 0`. By default, search result files are overwritten at the start of a run (`overwrite_results: true`) so repeated launches do not mix path IDs from old runs.

### Greedy Search

`greedy_search` is a deterministic non-neural baseline. It also goes through `dataset.test_file` one molecule at a time, matching evaluation order. At each step:

1. Build the valid template list using the configured masking mode.
2. Enumerate candidate products from valid templates.
3. For bimolecular templates, enumerate up to `max_r2_per_template` valid second reactants.
4. Score candidates with the configured reward mode.
5. Choose the candidate with the best score and continue from that product.

For `reward: delta_qed`, greedy search maximizes QED improvement at each step. For `reward: final_qed`, it maximizes product QED.

Greedy search supports two modes:

```yaml
search:
  greedy_mode: best_action          # always take the highest-scoring valid action
  # greedy_mode: positive_delta_only # for delta_qed, stop if the best action is <= 0
```

Use the wrapper override:

```bash
GREEDY_MODE=positive_delta_only ./run_launcher/sh_files/run_genmolrl_greedy_search.sh
```

`best_action` may take a negative delta-QED action if every valid action is negative, choosing the least bad one. `positive_delta_only` stops instead, keeping the current molecule as the final molecule.

### Exhausted Search

`exhausted_search` is a deterministic non-neural baseline. It goes through the molecules in `dataset.test_file` one by one. For each start molecule, it recursively enumerates every valid next reaction under the configured masking mode. A trajectory is saved when either:

- there is no valid next action from the current molecule, or
- the trajectory reaches `max_episode_len`.

By default, the exhaustive config leaves `max_paths`, `max_reactions`, `max_starts`, and `max_r2_per_template` unset, so it attempts the full search space. This can become very large, especially for Bi mode (under bi `reaction_valid` masking, the average start molecule has on the order of 10² R1-feasible templates × 10³–10⁴ pattern-compatible R2 partners per template at depth 1 alone). Set those fields in the config for a bounded debug run.

Trajectories are streamed to the on-disk results file **per path** as the
search progresses (rather than buffered in RAM until the run finishes),
so progress is observable in `runs/exhausted_search_{uni,bi}_results.txt`
during long bi runs and the process can be safely interrupted without
losing completed paths. Run them with:

```bash
./run_launcher/sh_files/run_genmolrl_exhausted.sh       # uni
./run_launcher/sh_files/run_genmolrl_exhausted_bi.sh    # bi
```

Search stopping controls:

```yaml
max_episode_len: 5     # max reaction depth per trajectory, all methods
search:
  max_paths: null      # optional cap on saved trajectories
  max_attempts: null   # legacy/debug cap; random/greedy default to test-set order
  max_reactions: null  # optional cap on total reactions
  max_starts: null     # optional cap on test starts for debug runs
  greedy_mode: best_action
```

These search settings are local to non-neural search and do not affect PPO, A2C, or TD3 configs.

## Reaction Modes

`reaction_mode` controls which templates are available.

### `uni`

Uses only:

- `unimolecular`
- `unimolecular_explicit_reagent`

This is the mode used by the current PPO, A2C, and TD3 Uni runs.

### `bi`

Uses all available templates:

- `unimolecular`
- `unimolecular_explicit_reagent`
- `bimolecular`

For PPO/A2C, Bi has two action parameterisations. The single state input
in every case is the Morgan fingerprint of the current molecule, which is
`R1` for the next bi-reaction; below we write the conditioning on `R1`
directly to keep things unambiguous.

1. **`MultiDiscrete([T, R2])`** — `π(T | R1) · π(R2 | R1)` (default for
   `--algorithm ppo` + `algorithm_family: sb3_multidiscrete`). Template
   `T` and second reactant `R2` are separate categorical action heads
   sampled independently. **The R2 head does not see `T`** — both heads
   consume the shared trunk over `fp(R1)`, then T- and R2-axis masks are
   applied independently.

2. **Hierarchical `π(T | R1) · π(R2 | R1, T)`** (`--algorithm ppo_bi` +
   `ppo_bi.policy_arch: hierarchical`). The R2 head **does** see `T`: the
   trunk feature for `R1` is concatenated with a learned per-template
   embedding of `T` and passed through the R2 query MLP, which then
   dot-products against per-reactant embeddings to produce the R2 logits.
   The R2 mask is per-`(R1, T)` rather than per-`R1`. So the policy can
   express template-conditional R2 preferences and the R2 distribution
   shifts with `T`.

The same `ppo_bi` trainer also exposes `policy_arch: multidiscrete`,
which matches parameterisation (1) but lives in the hand-rolled trainer
so PPO accounting (GAE, clipped surrogate, value clipping, target_kl
early stop, explained-variance logging) is identical to GraphTransPPO
and source-of-truth comparable to the MaskablePPO Bi run.

For TD3/PGFS, Bi uses the PGFS decomposition:

```text
template selector -> continuous R2 fingerprint query -> KNN over valid second reactants
```

## Action Families

`algorithm_family` controls the environment action interface.

### `sb3_discrete`

Used by PPO/A2C Uni.

```text
Discrete(num_templates + 1)
```

The last action is Stop when `use_stop_action=true`.

### `sb3_multidiscrete`

Used by PPO/A2C Bi compatibility mode (`--algorithm ppo` with
`configs/ppo_bi_multidiscrete_delta_qed.yaml`).

```text
MultiDiscrete([num_templates + 1, num_reactants])
```

The first component is template/Stop. The second component is the R2 index. For uni templates, R2 is ignored.

### `ppo_bi_hierarchical`

Used by the hand-rolled BiPPO trainer (`--algorithm ppo_bi`). The trainer
*does not* construct a Gymnasium env; instead it consumes the same
`env.*` fields (`use_stop_action`, `invalid_reaction_penalty`,
`stop_early_penalty`, `stop_penalty_until_step`, `reward_round_digits`,
`info_qed_round_digits`) and operates over the same SMILES action triple
`(state, T, R2)` as `MoleculeDesignEnv`, but the policy emits the joint
`(T, R2)` directly. State conditioning is exactly `s = fp(R1)` (1024-d
Morgan fingerprint of the current molecule, which becomes R1 of the next
reaction). Two architectures are supported via `ppo_bi.policy_arch`:

- `hierarchical`: autoregressive `π_T(T | R1) · π_R2(R2 | R1, T)`. The
  R2 query is built from the trunk encoding of `R1` concatenated with a
  learned per-template embedding of `T` (`template_embed_dim=64` by
  default), then projected to a `r2_embed_dim=64` query that
  dot-products against per-reactant embeddings. So the R2 logits are a
  function of **both R1 and T**.
- `multidiscrete`: independent `π_T(T | R1) · π_R2(R2 | R1)`. The R2
  query consumes only the trunk encoding of `R1`; `T` is dropped from
  the R2 head.

#### `ppo_bi.r2_arch` — R2 representation

How the per-reactant embeddings are produced is controlled by the
`r2_arch` field on the trainer config block (same knob for both
`policy_arch` values):

- `lookup` (legacy): a learned `nn.Embedding(num_reactants_train,
  r2_embed_dim)` table. The R2 vocabulary is **baked at training
  time**; the embedding rows have no meaningful values for any other
  pool, so `evaluate()` is forced to draw R2 from the same train pool.
  Bit-identical to the original Bi-PPO runs.
- `encoder` (current default in `configs/ppo_bi_*.yaml`): an MLP
  encoder over Morgan fingerprints of the candidate R2 SMILES,
  `r2_keys = MLP_R2(fp(R2_SMILES))`. The encoder weights are shared
  between train and test pools, so `evaluate()` can swap the active R2
  pool to `data/Bi/reactants_test.pkl` and the policy never sees train
  R2s at eval time — which is the only way to get "strict test-only R2
  at evaluation" given the disjoint bi train/test reactant pools.

##### R2 encoder capacity (Option 1 upgrade for Bi-PPO)

Under `r2_arch: encoder` the encoder MLP itself has two variants,
selected by `r2_encoder_residual`:

- `r2_encoder_residual: false` (legacy): a plain 2-layer MLP
  `1024 → r2_encoder_hidden → r2_embed_dim`. ~558k parameters at
  the default `r2_encoder_hidden=512`.
- `r2_encoder_residual: true` (current default in
  `configs/ppo_bi_*.yaml`): a deeper Pre-LN residual MLP. Layout:
  `Linear(1024 → r2_encoder_hidden) → LayerNorm → ReLU` for the
  stem, then `r2_encoder_n_res_blocks` residual blocks (each
  `LayerNorm → Linear → ReLU → Linear → +skip`) at width
  `r2_encoder_hidden`, then a `Linear(r2_encoder_hidden →
  r2_embed_dim)` projection. With the defaults
  `r2_encoder_hidden: 1024` and `r2_encoder_n_res_blocks: 2` that's
  ~5.3M parameters — roughly 10× the legacy encoder, but the
  LayerNorm + skip backbone keeps the deeper stack stable under PPO
  advantage noise. Use this variant unless you're trying to
  reproduce an older Bi-PPO baseline.

### `graphtransppo`

Used by GraphTransPPO (`--algorithm graphtransppo`). Action space is
`Discrete(num_templates + 1)` like `sb3_discrete`, but the env builds a
PyG graph observation (`reaction_graph_action` design) so the policy and
value heads can be applied on the graph-transformer trunk directly. Bi
support is provided separately by `graphtransppo_bi` (below).

### `graphtransppo_bi`

Used by Bi-GraphTransPPO (`--algorithm graphtransppo_bi`). Same hand-rolled
trainer family as `ppo_bi` — supports both `policy_arch: hierarchical` and
`policy_arch: multidiscrete` with identical PPO accounting — but the R1
trunk is replaced by a GraphTransformer over the molecular graph instead
of an MLP over Morgan FPs.

#### `graphtransppo_bi.r2_arch` — R2 representation

Three variants, selected by `r2_arch`:

- `lookup`: same legacy `nn.Embedding(num_reactants_train, r2_embed_dim)`
  table as `ppo_bi.r2_arch: lookup`.
- `encoder`: same Morgan-FP MLP encoder as `ppo_bi.r2_arch: encoder`. The
  R1 tower is a GraphTransformer; the R2 tower is an MLP over
  fingerprints — an **asymmetric** two-tower retrieval head.
- `encoder_graph` (current default, Option 3 upgrade): a **Siamese**
  R2 GraphTransformer + linear projection. Both towers are
  graph-encoded; the R2 input is the molecular graph itself, not a
  fingerprint. The R2 backbone has **separate weights** from the R1
  backbone — no weight tying — so each tower can be sized
  independently:

  - R1 backbone: full GraphTransformer (`num_emb: 64`, `num_layers:
    3`, `num_heads: 2`) — the trunk shared by template / value /
    R2-query heads.
  - R2 backbone: smaller GraphTransformer (`r2_num_emb: 32`,
    `r2_num_layers: 1`, `r2_num_heads: 2` by default) followed by a
    `Linear(2 * r2_num_emb → r2_embed_dim)` projection. The R2 pool
    is re-encoded over ~116k candidate graphs during each PPO update,
    so the R2 side is intentionally smaller to keep wall-clock cost
    manageable.

  Because the Siamese R2 encoder is expensive per call, the trainer
  caches its output with an explicit refresh cadence:
  `r2_keys_refresh_minibatches` (default `8`). On the first PPO
  minibatch of each update cycle (and every K minibatches thereafter)
  the R2 pool is re-encoded *with gradients*; on intermediate
  minibatches the trainer reuses a detached copy of the previous
  fresh encoding. The R1 trunk, template head, value head, and R2
  query head still get gradient every minibatch — only the Siamese
  R2 backbone's gradient signal is downsampled. Set
  `r2_keys_refresh_minibatches: 1` to recover per-minibatch R2
  gradients at the cost of ~K× more pool encodings per update.

### `td3_pgfs`

Used by custom TD3/PGFS.

The vector passed to `env.step` is always `(template_one_hot, r2_placeholder_tensor)`. Gym’s declared space remains **`Discrete(num_templates + stop)`** for compatibility; the placeholder’s width depends on **`env.action_design`**:

#### `pgfs_continuous_r2` (continuous PGFS-style head)

Default for uni TD3 configs that compare against the full PGFS parameterization.

- **Actor:** template logits + **continuous R2 head** (output dim = Morgan FP length, same as `observation_space` without appended masks). Uni templates never use the R2 output (zeros); bi templates feed KNN.
- **Critic:** `Q(state, template_one_hot, r2_vector)` with that full R2 width.
- **Rollout:** wrapped in **`KNNWrapper`**, which maps the continuous R2 vector to a discrete pool SMILES for bimolecular templates.

#### `td3_uni_discrete` (discrete template-only TD3)

For **`reaction_mode: uni`** comparison runs only.

- **Actor:** **template logits only** (no Pi / R2 network). The second tensor in the step tuple has shape **`(0,)`** (empty).
- **Critic:** `Q(state, template_one_hot)` — no R2 channels in the network or replay buffer.
- **Rollout:** **no `KNNWrapper`** (unnecessary when all templates are uni-molecular).

Masking (`none`, `substructure`, `reaction_valid`, `r2_available`) is unchanged and controlled by the top-level **`masking`** field for both modes.

Example configs: `configs/td3_uni_continuous_masked_delta_qed.yaml` (continuous) vs `configs/td3_uni_discrete_masked_delta_qed.yaml` (discrete).

Use **`run_genmolrl_td3_continuous.sh`** or **`run_genmolrl_td3_discrete.sh`** for experiment setup. Both delegate to **`run_genmolrl_td3_common.sh`**, which sources the shared launcher prelude and invokes the unified `run_experiment` CLI. All hyperparameters — `project`, `experiment_name`, `env.action_design`, `masking`, `reward`, and the `td3.*` block — come from each runner's default YAML (`configs/td3_uni_continuous_masked_delta_qed.yaml` and `configs/td3_uni_discrete_masked_delta_qed.yaml`). Override the YAML path via `CONFIG=...`, or override a single field via `MASKING=...`, `REWARD=...`, `REACTION_MODE=...`, `EXPERIMENT_NAME=...`. The launchers themselves never inject defaults.

## Masking Modes

Masking controls which template actions are considered legal before the policy samples/selects an action. Stop is appended separately when enabled.

Each masking mode comes with a contract about whether `apply_reaction`
failures (RDKit kekulisation / sanitisation pathologies, or — for bi
templates — no valid R2 pairing) are allowed to surface as
`invalid_reaction_penalty` (-1) at env-step time. The summary table:

| Mask | Template-axis check | R2-axis check (bi only) | Runs `apply_reaction` at mask time? | `-1` allowed at step time? |
|---|---|---|---|---|
| `none` | none | none | no | yes |
| `substructure` | R1 SMARTS match | none | no | yes |
| `r2_available` | R1 SMARTS match | ≥1 R2 in pool pattern-matches the R2 slot | no | yes |
| `reaction_valid` | R1 match + RDKit produces a sanitised product (uni: with R2=None; bi: with some pool R2) | the per-(state, T) RDKit-validated R2 set (for bi-aware consumers) | yes (expensive for bi) | **no** |

The mask kind is selected by the top-level `masking` field in the YAML
and overridable from the wrapper scripts via `MASKING=<kind>`.

### `none`

No template validation at mask time.

```text
mask[i] = 1 for every template
```

Invalid reactions may still fail in `env.step()` and receive `invalid_reaction_penalty`.

### `substructure`

A template is valid iff the current molecule matches the first-reactant
SMARTS pattern. Uni and bi templates are treated the same way — the bi
case does **not** add an R2 availability check (that's `r2_available`'s
job).

Validation:

```text
R1 has RDKit substructure match to reaction reactant template 0
```

Implementation uses:

```python
mol.HasSubstructMatch(reaction.GetReactantTemplate(0), useChirality=True)
```

This does not run the reaction. A template can pass this mask and still
fail at step time when `apply_reaction` cannot sanitise the product (or,
for bi templates, when the sampled R2 is structurally incompatible with
the R2 SMARTS slot). Those failures are recorded as `-1` transitions so
the policy learns from the failure signal.

### `r2_available`

A template is valid if:

1. The current molecule matches the first-reactant SMARTS pattern.
2. If the template is unimolecular, the template is valid.
3. If the template is bimolecular, at least one reactant in the pool
   matches the second-reactant SMARTS pattern.

Validation:

```text
uni valid = first-reactant match
bi valid = first-reactant match AND at least one valid R2 exists
```

This does not run any full `R1 + R2 -> product` reaction at mask time;
the R2 check is a SMARTS pattern match, not a sanitisation run. Pattern
matching does *not* guarantee a sanitised product, so `apply_reaction`
may still fail at step time — those rare failures are recorded as `-1`
transitions, identical to `substructure`.

`r2_available` is the PGFS/TD3-style feasibility check (TD3 picks R2
later via the continuous KNN mechanism), and is also the default Bi-PPO
masking — cheap to compute, and the residual `-1` rate is a useful
training signal.

### `reaction_valid`

A template is valid if:

- **Uni templates** (`unimolecular`, `unimolecular_explicit_reagent`):
  1. The current molecule matches the first-reactant SMARTS pattern.
  2. `apply_reaction(R1, template, None)` returns a sanitised product
     SMILES.
- **Bi templates** (`bimolecular`):
  1. The current molecule matches the first-reactant SMARTS pattern.
  2. There exists some `R2` in `dataset.training_file` (the reactant
     pool) such that `apply_reaction(R1, template, R2)` returns a
     sanitised product. `R2=None` is tried first so bi templates that
     bake fixed reagents into `_explicit_reagents` short-circuit without
     scanning the pool.

Validation:

```text
uni: first-reactant match AND apply_reaction(state, T, None) returns a sanitized product
bi:  first-reactant match AND ∃ R2 in pool s.t. apply_reaction(state, T, R2) returns a sanitized product
```

The uni branch is the exact original PPO/A2C Uni masking behavior from
`exp_branch_run_PPO_mask_extendedObs.sh` and `exp_branch_run_A2C_mask.sh`
(in uni mode the manager only holds uni-type templates, so the bi branch
is unreachable and the mask is bit-identical to the pre-Bi-fix
implementation).

The bi branch is the strict counterpart used by Bi-PPO when zero `-1`
rewards are required:

- **Hierarchical Bi-PPO** (`ppo_bi.policy_arch: hierarchical`) reads the
  per-(state, T) RDKit-validated R2 mask from
  `ReactionManager.bi_r2_valid_mask(state, t)` and is therefore zero-`-1`
  by mask construction; no rejection sampling is needed.
- **Multidiscrete Bi-PPO** (`ppo_bi.policy_arch: multidiscrete`) samples
  T and R2 independently from a per-state R2 union mask, which can pair
  an R2 with a template it doesn't actually work for. To keep the
  zero-`-1` contract, the trainer runs online joint rejection sampling
  (`ppo_bi.r2_resample_retries` retries; budget exhaustion falls through
  to STOP rather than violate the contract).

The bi-mask cost is non-trivial: each cache-cold `(state, T)` pays one
`apply_reaction` per pattern-matched R2 candidate. The values are cached
per `(state, T)` and per `(state, "reaction_valid")` template mask, so
amortised cost is fine for long training runs but can dominate quick
smoke tests. Use `r2_available` or `substructure` if the `-1` signal is
acceptable.

## Default Masking By Algorithm

The current default configs use:

```text
PPO Uni:                       reaction_valid
A2C Uni:                       reaction_valid
GraphTransRL Uni:              reaction_valid
GraphTransPPO Uni:             reaction_valid
TD3 Uni:                       r2_available
PPO Bi MultiDiscrete (SB3):    r2_available    (override: MASKING=reaction_valid)
PPO Bi Hierarchical (BiPPO):   r2_available    (override: MASKING=reaction_valid → zero -1)
Random / Greedy / Exhausted (Uni/Bi):  reaction_valid (config-driven; can be overridden)
```

PPO/A2C/GraphTrans* use `reaction_valid` for uni runs to exactly match the
original experiments-branch Uni runs and to keep the policy's training
signal clean of `-1` outliers.

TD3 uses `r2_available` to preserve PGFS-style template feasibility and
keep R2/KNN handling separate.

Bi-PPO defaults to `r2_available` for wallclock-cheap training; flip to
`reaction_valid` (e.g.
`MASKING=reaction_valid ./run_launcher/sh_files/run_genmolrl_ppo_bi_hierarchical.sh`)
when zero `invalid_reaction_penalty` rewards in recorded transitions is
a requirement. The same override on the multidiscrete launcher uses the
fixed bi-aware template mask but does not enforce joint `(T, R2)`
validity at SB3 sampling time.

## Reward Modes

### `delta_qed`

Reward is the change in QED from the previous molecule to the new molecule:

```text
reward = QED(product) - QED(previous)
```

This is the default objective for the current PPO, A2C, and TD3 experiments.

For PPO/A2C compatibility, the default configs round rewards to 3 decimals, matching the old environment.

### `final_qed`

Reward is the QED of the product molecule:

```text
reward = QED(product)
```

This is useful when the objective is absolute final molecule quality rather than improvement from the start molecule.

## Start Strategies

### `random_pool`

Training episodes start from a random molecule in the reactant pool.

This matches current PPO/A2C/TD3 training behavior.

### `cycle_file`

Episodes cycle through a SMILES file deterministically.

### `cycle_pool`

Episodes cycle through every molecule in the loaded reactant pool deterministically.

This matches current evaluation behavior: the eval environment loads `dataset.test_file`, resets the cycle to the first molecule at each evaluation pass, and runs one episode for every test molecule.

### `fixed`

Every episode starts from the same molecule. Useful for debugging.

### `learned_policy`

Reserved for SynFlowNet-style first-reactant selection, where the model learns to pick the first building block as part of the trajectory.

## Stop Action

For PPO/A2C, Stop is the final discrete template action when `use_stop_action=true`.

The Stop reward is controlled by:

```yaml
stop_early_penalty: 0.0
stop_penalty_until_step: 3
```

For the PPO/A2C Uni compatibility configs, this matches the original experiments-branch setup: early Stop is allowed and has zero penalty.

TD3 Uni uses `reward: delta_qed` with `use_stop_action: true`. The
conditional Stop penalty (`stop_early_penalty`, `stop_penalty_until_step`)
is wired in `rewards.stop_reward` and `molecule_design_env.step` and only
fires when `stop_penalty_until_step > 0`, `current_step <=
stop_penalty_until_step`, *and* at least one feasible reaction template
was available at that state. It is left **disabled by default**
(`stop_early_penalty: 0.0`, `stop_penalty_until_step: -1`) in the TD3
YAMLs while we evaluate the impact of the May 2026 audit fixes
(action mask in observation, smaller replay buffer); the conditional
code path remains in place for future experiments. See "TD3/PGFS → Uni
`delta_qed` convergence ceiling — diagnosis & fix history" for the full
write-up of every lever tried.

`td3.warmup_stop_probability` is preserved for compatibility but is set
to `0.0` in the current YAMLs so warmup samples templates exclusively
(letting the replay buffer accumulate template transitions before the
actor starts updating).

## Logging

GenMolRL defines PPO-compatible W&B metrics against:

```text
train/global_step
```

Important metric names include:

```text
training/total_reward_each_episode
train/mean_reward
eval/mean_reward
episode_length
reward_per_step
qed_per_step
overall_max_qed
```

Bi-PPO additionally exposes:

```text
train/invalid_reaction_count        # cumulative -1 transitions
train/rollout_invalid_count_cum     # snapshot per rollout
train/rejection_count_cum           # joint-rejection retries (multidiscrete + reaction_valid only)
train/stop_event_count              # cumulative STOP actions chosen
train/rollout_stop_fraction         # STOP fraction per rollout
```

Under `masking=reaction_valid` the two `invalid_*` counters must stay at
0 throughout training (the contract is enforced by an `assert` in the
rollout). Under `substructure` / `r2_available` they grow naturally —
those modes deliberately allow `apply_reaction` failures to surface as
`-1` transitions for the policy to learn from.

Outputs are written under:

```text
runs/<run_id>/
```

Bi exhaustive search additionally writes streaming results to
`runs/exhausted_search_bi_results.txt` (per-path append; safe to tail
during a run and safe to interrupt without losing completed paths).

## Compatibility Notes

- PPO/A2C Uni `reaction_valid` masks were directly compared against the original experiments-branch `ReactionManager` and matched on sampled starts.
- Template insertion order is preserved from the pickle so action indices match legacy behavior.
- PPO/A2C reward and info QED rounding are configurable and default to 3 decimals in compatibility configs.
- TD3 reuses the existing custom PGFS TD3 agent/replay/KNN implementation through adapters while GenMolRL owns the config, environment, staging, and launcher.
- The bi-branch of `ReactionManager.template_reaction_valid_mask` is uni-unreachable: in `reaction_mode: uni` only uni-type templates are registered, so the uni-mode `reaction_valid` mask is bit-identical to the pre-Bi-fix implementation. Uni runs are therefore behaviour-compatible with all earlier PPO/A2C numbers.
- GraphTransPPO and Bi-PPO are new trainers (not migrations); they reuse
  the GenMolRL `ReactionManager`, masking, reward, dataset staging, and
  W&B logging stack, but their PPO accounting is hand-rolled rather than
  delegated to SB3 so the rollout/eval cadence and explained-variance
  logging are directly source-of-truth comparable across the two.

## Smoke Checks

After staging data, run from the project root (`<repo>/GenMolRL`):

```bash
PYTHONPATH=. python -m compileall -q genmolrl
PYTHONPATH=. python - <<'PY'
from tests.test_env_smoke import test_ppo_uni_env_reset, test_td3_uni_env_reset
test_ppo_uni_env_reset()
test_td3_uni_env_reset()
print("smoke ok")
PY
```
