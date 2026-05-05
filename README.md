# GenMolRL

GenMolRL is the unified molecule-generation project for the current PPO, A2C, and TD3/PGFS experiments. It replaces the previous workflow where PPO/A2C and TD3 were launched from different branch directories with duplicated environment, masking, data staging, reward, and logging logic.

The first goal is behavior-compatible migration of:

- `exp_branch_run_PPO_mask_extendedObs.sh`
- `exp_branch_run_A2C_mask.sh`
- `run_td3.sh`
- `run_random_search.sh`
- legacy greedy search scripts under `designing-new-molecules/src/training/`

The second goal is to provide a structured place for future methods such as SynFlowNet and REINVENT scaffold decorator.

## Layout

```text
GenMolRL/
  genmolrl/
    chem/          # RDKit reactions, fingerprints, product selection, dataset staging
    envs/          # unified molecule-design Gymnasium env, masks, rewards, starts
    algorithms/    # PPO, A2C, TD3/PGFS trainers
                   # plus random and greedy search baselines
    logging/       # W&B metrics and callbacks
    methods/       # future SynFlowNet / scaffold-decorator adapters
    scripts/       # unified CLI entry points
  configs/         # experiment YAML configs
  tests/           # smoke tests
```

## Installation

GenMolRL currently uses the repository-level conda environment exported at:

```text
/home/bz317/new_mol_with_RL_GNN/conda_env_RL_for_new_mol.yml
```

Create the environment from scratch:

```bash
cd /home/bz317/new_mol_with_RL_GNN
conda env create -f conda_env_RL_for_new_mol.yml
conda activate RL_for_new_mol
```

If the environment already exists and you want to update it from the file:

```bash
cd /home/bz317/new_mol_with_RL_GNN
conda env update -n RL_for_new_mol -f conda_env_RL_for_new_mol.yml --prune
conda activate RL_for_new_mol
```

The environment includes the main dependencies used by the current experiments, including RDKit, PyTorch, Stable-Baselines3, sb3-contrib, W&B, and FAISS GPU packages. If you run on a machine without a compatible GPU/CUDA setup, TD3 can still import but FAISS/KNN or CUDA execution may fall back or need environment-specific adjustment.

GenMolRL is not installed as a site package by default. Run commands with `PYTHONPATH=GenMolRL` from the repository root:

```bash
cd /home/bz317/new_mol_with_RL_GNN
PYTHONPATH=GenMolRL python -m genmolrl.scripts.stage_data
```

Optional editable install:

```bash
cd /home/bz317/new_mol_with_RL_GNN/GenMolRL
python -m pip install -e .
```

After an editable install, `PYTHONPATH=GenMolRL` is usually no longer necessary, but the wrapper scripts still set it explicitly for portability.

For W&B logging, either log in normally:

```bash
wandb login
```

or place the API key in the existing repository-level file:

```text
/home/bz317/new_mol_with_RL_GNN/wandb_api_key.txt
```

The wrapper scripts read this file automatically. To disable cloud logging for smoke tests:

```bash
WANDB_MODE=disabled ./run_genmolrl_ppo.sh
```

Quick installation check:

```bash
cd /home/bz317/new_mol_with_RL_GNN
conda activate RL_for_new_mol
PYTHONPATH=GenMolRL python -m compileall -q GenMolRL/genmolrl
PYTHONPATH=GenMolRL python -m genmolrl.scripts.stage_data
```

## Data Staging

The current Uni experiments use the same staged files as the old scripts, copied from:

```text
designing-new-molecules/data/train_test_split/uni_molecular/
```

Stage them with:

```bash
PYTHONPATH=GenMolRL python -m genmolrl.scripts.stage_data
```

This creates the canonical Uni directory:

```text
GenMolRL/data/Uni/reactants_train.pkl
GenMolRL/data/Uni/reactants_val.pkl
GenMolRL/data/Uni/templates_unimolecolar_explicit.pkl
```

It also creates derived compatibility files in the same directory:

```text
GenMolRL/data/Uni/reactants_full.pkl
GenMolRL/data/Uni/eval_start_smiles.txt
```

`reactants_full.pkl` is the merged train+val reactant pool used by the compatibility PPO/A2C/TD3 configs. `eval_start_smiles.txt` contains the validation SMILES and is used for deterministic evaluation starts.

The future Bi dataset should live under:

```text
GenMolRL/data/Bi/
```

That directory is intentionally empty for now. The spelling `unimolecolar` is preserved because existing data and scripts use that filename.

## Running Experiments

Use the wrapper scripts from the repository root:

```bash
./run_genmolrl_ppo.sh
./run_genmolrl_a2c.sh
./run_genmolrl_td3.sh
./run_genmolrl_random_search.sh
./run_genmolrl_greedy_search.sh
```

Or call the unified launcher directly:

```bash
PYTHONPATH=GenMolRL python -m genmolrl.scripts.run_experiment \
  --algorithm ppo \
  --reaction-mode uni \
  --masking reaction_valid \
  --reward delta_qed \
  --config GenMolRL/configs/ppo_uni_masked_delta_qed.yaml
```

Common environment overrides:

```bash
EXPERIMENT_NAME=PPO_Uni_test ./run_genmolrl_ppo.sh
MASKING=none ./run_genmolrl_ppo.sh
REWARD=final_qed ./run_genmolrl_a2c.sh
REACTION_MODE=bi ./run_genmolrl_td3.sh
WANDB_MODE=disabled ./run_genmolrl_random_search.sh
WANDB_MODE=disabled ./run_genmolrl_ppo.sh
```

## Supported Algorithms

### PPO

`ppo` trains a policy with Stable-Baselines3 / sb3-contrib. Masked mode uses `MaskablePPO`; no-mask mode uses plain SB3 PPO.

### A2C

`a2c` trains an actor-critic policy. Masked mode uses a custom policy that reads the action mask from the observation tail; no-mask mode uses plain `MlpPolicy`.

### TD3/PGFS

`td3` trains the custom PGFS-style TD3 implementation. It learns a template selector plus a continuous R2 vector for bimolecular reactions.

### Random Search

`random_search` is a non-neural baseline. At each step:

1. Select a random starting molecule from the configured reactant pool.
2. Build the valid template list using the configured masking mode.
3. Randomly choose one valid template.
4. If the template is bimolecular, randomly choose one valid R2 from the reactant pool.
5. Apply the reaction and continue from the product if valid.

It writes a `.txt` report containing a summary section plus successful synthesis steps, and logs simple W&B metrics such as saved paths, total reactions, last terminal QED, and best QED.

The text report includes a `START` row for each saved path (`step=0`) followed by one row per successful reaction. By default, search result files are overwritten at the start of a run (`overwrite_results: true`) so repeated launches do not mix path IDs from old runs.

### Greedy Search

`greedy_search` is a non-neural baseline. At each step:

1. Build the valid template list using the configured masking mode.
2. Enumerate candidate products from valid templates.
3. For bimolecular templates, enumerate up to `max_r2_per_template` valid second reactants.
4. Score candidates with the configured reward mode.
5. Choose the candidate with the best score and continue from that product.

For `reward: delta_qed`, greedy search maximizes QED improvement at each step. For `reward: final_qed`, it maximizes product QED.

Search stopping controls:

```yaml
search:
  max_steps: 5        # max reaction depth per path
  max_paths: 100      # max saved successful paths
  max_attempts: 1000  # max attempted starts
  max_reactions: 10000
```

These search settings are local to random/greedy search and do not affect PPO, A2C, or TD3 configs.

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

The learned action is:

```text
(template_one_hot, r2_vector)
```

For Uni templates, R2 is a zero/no-op vector. For Bi templates, the continuous R2 vector is converted to a discrete second reactant through KNN.

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

This matches current evaluation behavior, where `eval_start_smiles.txt` contains validation molecules and evaluation cycles through them.

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

TD3/PGFS currently defaults to `use_stop_action: false` in its compatibility config, following the original PGFS-style action decomposition.

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
GenMolRL/runs/<run_id>/
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
