# Search QED By Step Visualization

`plot_search_qed_by_step.py` compares exhaustive, greedy, and random search.

The generated plot has a no-action column plus up to five reaction columns.
Column `N` shows the QED distribution of final molecules from trajectories that
ended after exactly `N` reactions. Reaction columns contain boxplots for
exhaustive, greedy, random, PPO, and A2C.

The no-action column contains exhaustive-search starts whose original molecule
has QED greater than or equal to every reachable action point. Greedy and random
do not currently save a no-action decision, so they are not shown in that
column.

The `n` annotation under each boxplot is the number of QED values used for that
box. The values differ across methods and step counts because not every start
molecule produces a trajectory of every length.

For exhaustive search, the script first finds the single best available action
point for each start molecule across all saved paths and steps 1-5. That best
QED contributes to exactly one step column. If the original start molecule has
QED greater than or equal to every reachable action point, the start contributes
to the no-action column.

Greedy and random search currently save only one final trajectory per start
molecule, so their step column contains starts whose single trajectory ended
after exactly that many reactions.

PPO and A2C are evaluated by loading the saved SB3 models and rolling them out
once over the test set with deterministic actions. The default model paths are:

- PPO: `runs/3zd846ff/wandb_model/model.zip`
- A2C: `runs/en3i9xg8/wandb_model/model.zip`

Their evaluated trajectories are cached so rerunning the plot does not need to
reload the models and re-evaluate all test molecules every time:

- `ppo_3zd846ff_eval_trajectories.csv`
- `a2c_en3i9xg8_eval_trajectories.csv`

Outputs:

- `search_qed_by_step_boxplots.png`
- `search_qed_by_step_summary.csv`
- `search_qed_exhaustive_best_action_stats.txt`

Regenerate them from the inner `GenMolRL` repository directory:

```bash
python visualization/plot_search_qed_by_step.py
```
