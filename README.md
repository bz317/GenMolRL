# GenMolRL

GenMolRL is the unified molecule-generation project for the current PPO, A2C, TD3/PGFS, and GraphTransRL experiments. It replaces the previous workflow where PPO/A2C and TD3 were launched from different branch directories with duplicated environment, masking, data staging, reward, and logging logic.

The first goal is behavior-compatible migration of:

- `exp_branch_run_PPO_mask_extendedObs.sh`
- `exp_branch_run_A2C_mask.sh`
- `run_td3.sh`
- Random Search, Greedy Search, and Exhausted Search baselines
- GraphTransRL with a graph-transformer policy backbone

The second goal is to provide a structured place for future methods such as REINVENT scaffold decorator.

## Layout

```text
GenMolRL/
  genmolrl/
    chem/          # RDKit reactions, fingerprints, product selection, dataset staging
    envs/          # unified molecule-design Gymnasium env, masks, rewards, starts
    algorithms/    # PPO, A2C, TD3/PGFS, SynFlowNet trainers
                   # plus random and greedy search baselines
    logging/       # W&B metrics and callbacks
    methods/       # lazy method adapters
    scripts/       # unified CLI entry points
  configs/         # experiment YAML configs
  tests/           # smoke tests
```

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

GenMolRL is not installed as a site package by default. Run commands with `PYTHONPATH=GenMolRL` from the repository root:

```bash
cd <repo-root>
./run_genmolrl_ppo.sh
```

Optional editable install:

```bash
cd <repo-root>/GenMolRL
python -m pip install -e .
```

After an editable install, `PYTHONPATH=GenMolRL` is usually no longer necessary, but the wrapper scripts still set it explicitly for portability.

For W&B logging, either log in normally:

```bash
wandb login
```

or place the API key in the existing repository-level file:

```text
wandb_api_key.txt
```

The wrapper scripts read this file automatically. To disable cloud logging for smoke tests:

```bash
WANDB_MODE=disabled ./run_genmolrl_ppo.sh
```

Quick installation check:

```bash
cd <repo-root>
conda activate RL_for_new_mol
PYTHONPATH=GenMolRL python -m compileall -q GenMolRL/genmolrl
```

## Data

GenMolRL is self-contained when the following files are present:

```text
data/Uni/reactants_train.pkl
data/Uni/reactants_test.pkl
data/Uni/templates_unimolecolar_explicit.pkl
```

Check the staged files with:

```bash
PYTHONPATH=GenMolRL python -m genmolrl.scripts.stage_data
```

The command above only validates and reports the existing `data/Uni` layout. It does not depend on the old source tree. If you intentionally want to regenerate the staged files from an external split, pass that source explicitly:

```bash
PYTHONPATH=GenMolRL python -m genmolrl.scripts.stage_data --source-dir <external-uni-split-dir>
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

Use the wrapper scripts from the repository root:

```bash
./run_genmolrl_ppo.sh
./run_genmolrl_a2c.sh
./run_genmolrl_td3_continuous.sh   # uni TD3, PGFS continuous R2 head
./run_genmolrl_td3_discrete.sh      # uni TD3, template-only critic
./run_genmolrl_td3.sh               # alias → continuous (backward compatible)
./run_genmolrl_random_search.sh
./run_genmolrl_greedy_search.sh
./run_genmolrl_exhausted_search.sh
./run_genmolrl_graphtransrl.sh
```

Or call the unified launcher directly:

```bash
PYTHONPATH=GenMolRL python -m genmolrl.scripts.run_experiment \
  --algorithm ppo \
  --reaction-mode uni \
  --masking reaction_valid \
  --reward delta_qed \
  --config configs/ppo_uni_masked_delta_qed.yaml
```

Common environment overrides:

```bash
EXPERIMENT_NAME=PPO_Uni_test ./run_genmolrl_ppo.sh
MASKING=none ./run_genmolrl_ppo.sh
REWARD=final_qed ./run_genmolrl_a2c.sh
REACTION_MODE=bi ./run_genmolrl_td3_continuous.sh    # optional; overrides YAML only when set
WANDB_PROJECT=MyTeam EXPERIMENT_NAME=my_td3_run ./run_genmolrl_td3_continuous.sh
CONFIG=configs/td3_uni_masked_balance_delta_qed.yaml ./run_genmolrl_td3_continuous.sh   # override YAML path only when needed
MAX_EPISODE_LEN=3 ./run_genmolrl_ppo.sh
WANDB_MODE=disabled ./run_genmolrl_random_search.sh
WANDB_MODE=disabled ./run_genmolrl_ppo.sh
WANDB_MODE=disabled ./run_genmolrl_graphtransrl.sh
```

Search runners also accept dataset path overrides:

```bash
TEST_FILE=data/Uni/reactants_test.pkl \
TEMPLATE_FILE=data/Uni/templates_unimolecolar_explicit.pkl \
./run_genmolrl_random_search.sh
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

### GraphTransRL

`graphtransrl` trains a GenMolRL-owned graph-transformer RL policy. The method does not learn the first reactant: training episodes start from random molecules sampled from `dataset.training_file`, and each eval pass cycles through every molecule in `dataset.test_file` exactly once.

The policy backbone is a graph transformer with PyG `GENConv` and `TransformerConv` layers. Action logits cover reaction templates plus Stop; there is no exposed `AddFirstReactant` action path in GenMolRL. Rewards are per-action `delta_qed`, and `eval/mean_reward` is the mean summed per-action delta-QED over all test starts. The eval logs also include `eval/avg_delta_qed`, `eval/mean_final_delta_qed`, `eval/max_qed`, `eval/mean_ep_length`, and `eval/n_molecules`.

Run it with:

```bash
./run_genmolrl_graphtransrl.sh
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
GREEDY_MODE=positive_delta_only ./run_genmolrl_greedy_search.sh
```

`best_action` may take a negative delta-QED action if every valid action is negative, choosing the least bad one. `positive_delta_only` stops instead, keeping the current molecule as the final molecule.

### Exhausted Search

`exhausted_search` is a deterministic non-neural baseline. It goes through the molecules in `dataset.test_file` one by one. For each start molecule, it recursively enumerates every valid next reaction under the configured masking mode. A trajectory is saved when either:

- there is no valid next action from the current molecule, or
- the trajectory reaches `max_episode_len`.

By default, the exhaustive config leaves `max_paths`, `max_reactions`, `max_starts`, and `max_r2_per_template` unset, so it attempts the full search space. This can become very large, especially for Bi mode. Set those fields in the config for a bounded debug run.

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

For PPO/A2C, Bi currently preserves the requested old-style factorized action design:

```text
MultiDiscrete([T, R2])
```

This means template `T` and second reactant `R2` are separate categorical action heads. The R2 choice is not conditioned on the sampled template inside the policy distribution. This is preserved for compatibility and should be treated as a baseline implementation, not the final ideal Bi action design.

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

Used by PPO/A2C Bi compatibility mode.

```text
MultiDiscrete([num_templates + 1, num_reactants])
```

The first component is template/Stop. The second component is the R2 index. For uni templates, R2 is ignored.

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

Use **`run_genmolrl_td3_continuous.sh`** or **`run_genmolrl_td3_discrete.sh`** for experiment setup: both export **`WANDB_PROJECT`** (default `GenMolRL`) and pass **`EXPERIMENT_NAME`** (defaults `TD3_Uni_Continuous` vs `TD3_Uni_Discrete`). **`run_genmolrl_td3.sh`** is a backward-compatible alias for the continuous runner. Hyperparameters—including **`env.action_design`**, **`masking`**, **`reward`**, and **`td3.*`**—come from each runner’s default YAML (override with **`CONFIG=...`**). Export **`REACTION_MODE`**, **`MASKING`**, or **`REWARD`** only when you intentionally override the config file.

## Masking Modes

Masking controls which template actions are considered legal before the policy samples/selects an action. Stop is appended separately when enabled.

### `none`

No template validation at mask time.

```text
mask[i] = 1 for every template
```

Invalid reactions may still fail in `env.step()` and receive `invalid_reaction_penalty`.

### `substructure`

A template is valid if the current molecule matches the first-reactant SMARTS pattern.

Validation:

```text
R1 has RDKit substructure match to reaction reactant template 0
```

Implementation uses:

```python
mol.HasSubstructMatch(reaction.GetReactantTemplate(0), useChirality=True)
```

This does not run the reaction. A template can pass this mask but still fail later if RDKit cannot generate a valid product.

### `reaction_valid`

A template is valid if:

1. The current molecule matches the first-reactant SMARTS pattern.
2. RDKit can run the reaction with no selected second reactant.
3. The first product sanitizes successfully.
4. A valid product SMILES is returned.

Validation:

```text
first-reactant match AND apply_reaction(R1, template, None) returns a sanitized product
```

This is the exact original PPO/A2C Uni masking behavior from `exp_branch_run_PPO_mask_extendedObs.sh` and `exp_branch_run_A2C_mask.sh`.

For true bimolecular templates, this usually returns invalid unless the template encodes a fixed explicit reagent, because no learned/selected R2 is supplied at mask time.

### `r2_available`

A template is valid if:

1. The current molecule matches the first-reactant SMARTS pattern.
2. If the template is unimolecular, the template is valid.
3. If the template is bimolecular, at least one reactant in the pool matches the second-reactant SMARTS pattern.

Validation:

```text
uni valid = first-reactant match
bi valid = first-reactant match AND at least one valid R2 exists
```

This does not run every full `R1 + R2 -> product` reaction at mask time. It is the PGFS/TD3-style feasibility check because TD3 chooses R2 later through the continuous R2/KNN mechanism.

## Default Masking By Algorithm

The current default configs use:

```text
PPO Uni: reaction_valid
A2C Uni: reaction_valid
TD3 Uni: r2_available
```

PPO/A2C use `reaction_valid` to exactly match the original experiments-branch Uni runs.

TD3 uses `r2_available` to preserve PGFS-style template feasibility and keep R2/KNN handling separate.

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

Outputs are written under:

```text
runs/<run_id>/
```

## Compatibility Notes

- PPO/A2C Uni `reaction_valid` masks were directly compared against the original experiments-branch `ReactionManager` and matched on sampled starts.
- Template insertion order is preserved from the pickle so action indices match legacy behavior.
- PPO/A2C reward and info QED rounding are configurable and default to 3 decimals in compatibility configs.
- TD3 reuses the existing custom PGFS TD3 agent/replay/KNN implementation through adapters while GenMolRL owns the config, environment, staging, and launcher.

## Smoke Checks

After staging data:

```bash
PYTHONPATH=GenMolRL python -m compileall -q GenMolRL/genmolrl
PYTHONPATH=".:GenMolRL" python - <<'PY'
from GenMolRL.tests.test_env_smoke import test_ppo_uni_env_reset, test_td3_uni_env_reset
test_ppo_uni_env_reset()
test_td3_uni_env_reset()
print("smoke ok")
PY
```
