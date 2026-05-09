"""Generate the GraphTransRL workflow diagram.

Produces ``graphtransrl_workflow.png`` (and ``.pdf``) in this directory. The
diagram shows the actual data flow used by ``genmolrl.algorithms.graphtransrl``:

    SMILES -> molecular graph -> GraphTransformer -> graph_embedding (128d)
           -> {template_head, stop_head} -> masked softmax -> sampled action
           -> RDKit reaction -> next SMILES (loop)
           -> trajectory + delta-QED rewards -> Trajectory Balance loss
                                              + best-K replay buffer
                                              + epsilon-greedy exploration

Run:
    python GenMolRL/workflow/plot_graphtransrl_workflow.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

COLORS = {
    "input":      "#cfe2ff",  # light blue   - molecular graph / data
    "encoder":    "#bfe3c5",  # light green  - GraphTransformer
    "head":       "#ffe5b4",  # light orange - MLP heads
    "action":     "#ffd6a5",  # peach        - action selection / mask
    "env":        "#e9ecef",  # gray         - environment / RDKit
    "loss":       "#f8c1c1",  # light red    - losses
    "buffer":     "#e0bbe4",  # purple       - replay buffer
    "border":     "#1f2937",
}

EDGE_LW = 1.4
BOX_LW = 1.3
FONT = {"family": "DejaVu Sans"}


def draw_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    color: str,
    fontsize: float = 10.0,
    title: str | None = None,
    title_fontsize: float = 11.0,
    text_color: str = "#111827",
):
    """Draw a rounded rectangle with an optional bold title and body text."""
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=BOX_LW,
        edgecolor=COLORS["border"],
        facecolor=color,
    )
    ax.add_patch(box)
    cx = x + w / 2
    if title is not None:
        ax.text(
            cx,
            y + h - 0.32,
            title,
            ha="center",
            va="center",
            fontsize=title_fontsize,
            fontweight="bold",
            color=text_color,
            **FONT,
        )
        ax.text(
            cx,
            y + (h - 0.55) / 2 + 0.05,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=text_color,
            **FONT,
        )
    else:
        ax.text(
            cx,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=text_color,
            **FONT,
        )


def arrow(
    ax,
    p0: tuple[float, float],
    p1: tuple[float, float],
    *,
    color: str = "#111827",
    style: str = "-|>",
    lw: float = EDGE_LW,
    rad: float = 0.0,
    text: str | None = None,
    text_offset: tuple[float, float] = (0.0, 0.18),
    text_fontsize: float = 9.0,
    ls: str = "-",
):
    a = FancyArrowPatch(
        p0,
        p1,
        arrowstyle=style,
        mutation_scale=14,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        linestyle=ls,
    )
    ax.add_patch(a)
    if text is not None:
        mx = (p0[0] + p1[0]) / 2 + text_offset[0]
        my = (p0[1] + p1[1]) / 2 + text_offset[1]
        ax.text(
            mx,
            my,
            text,
            ha="center",
            va="center",
            fontsize=text_fontsize,
            color=color,
            **FONT,
        )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def build_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(16.5, 9.0))
    ax.set_xlim(0, 16.5)
    ax.set_ylim(0, 9.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title.
    ax.text(
        7.75,
        8.55,
        "GraphTransRL workflow (uni-reactions, ΔQED reward, GFlowNet Trajectory Balance)",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#111827",
        **FONT,
    )

    # ------------------------------------------------------------------
    # Row 1 (top): SMILES -> graph features -> GraphTransformer -> graph emb
    # ------------------------------------------------------------------

    # 1. Current SMILES
    draw_box(
        ax,
        x=0.2,
        y=6.0,
        w=2.2,
        h=1.6,
        title="Current SMILES",
        text="s_t  (RDKit Mol)\nstart molecule provided\nby external sampler",
        color=COLORS["input"],
        fontsize=8.5,
    )

    # 2. Molecular graph features
    draw_box(
        ax,
        x=2.7,
        y=6.0,
        w=2.7,
        h=1.6,
        title="Molecular graph",
        text="nodes x : [N, 16]   atom 1-hot + Q, arom, H, deg\n"
             "edges e : [2E, 6]   bond 1-hot + conj, ring\n"
             "cond  c : [1, 1]    (currently set to 1)",
        color=COLORS["input"],
        fontsize=8.0,
    )

    # 3. GraphTransformer
    draw_box(
        ax,
        x=5.7,
        y=6.0,
        w=3.2,
        h=1.6,
        title="GraphTransformer (3 layers)",
        text="x→64, e→64, c→64 via MLPs\n"
             "+ virtual node per graph (global hub)\n"
             "GENConv → TransformerConv → AdaLN(c)",
        color=COLORS["encoder"],
        fontsize=8.0,
    )

    # 4. Graph embedding
    draw_box(
        ax,
        x=9.2,
        y=6.0,
        w=2.4,
        h=1.6,
        title="graph_embedding",
        text="[B, 128]\nconcat( mean_pool(node_h),\n         virtual_node_h )",
        color=COLORS["encoder"],
        fontsize=8.5,
    )

    # Arrows row 1.
    arrow(ax, (2.4, 6.8), (2.7, 6.8), text="smiles_to_data", text_offset=(0, 0.15))
    arrow(ax, (5.4, 6.8), (5.7, 6.8))
    arrow(ax, (8.9, 6.8), (9.2, 6.8))

    # ------------------------------------------------------------------
    # Row 2: heads
    # ------------------------------------------------------------------

    # 5a. template_head
    draw_box(
        ax,
        x=12.0,
        y=6.7,
        w=3.3,
        h=0.95,
        title="template_head (MLP 128→64→K)",
        text="ReactUni logits  [B, K=num_templates]",
        color=COLORS["head"],
        fontsize=8.5,
        title_fontsize=9.5,
    )

    # 5b. stop_head
    draw_box(
        ax,
        x=12.0,
        y=5.65,
        w=3.3,
        h=0.95,
        title="stop_head (MLP 128→64→1)",
        text="Stop logit  [B, 1]",
        color=COLORS["head"],
        fontsize=8.5,
        title_fontsize=9.5,
    )

    # arrows from graph embedding to heads
    arrow(ax, (11.6, 7.0), (12.0, 7.15), rad=0.05)
    arrow(ax, (11.6, 6.6), (12.0, 6.10), rad=-0.05)

    # ------------------------------------------------------------------
    # Row 3: action selection
    # ------------------------------------------------------------------

    # 6. Concatenate logits + mask + softmax
    draw_box(
        ax,
        x=10.7,
        y=4.0,
        w=4.6,
        h=1.2,
        title="Masked categorical policy",
        text="logits = concat([template_logits, stop_logit])  →  [B, K+1]\n"
             "infeasible templates → −1e9   (uses ReactionManager mask)\n"
             "π(·|s) = softmax(masked_logits)",
        color=COLORS["action"],
        fontsize=8.5,
        title_fontsize=10.0,
    )

    # arrows from heads down to softmax box
    arrow(ax, (12.8, 6.7), (12.8, 5.2), rad=0.0)
    arrow(ax, (13.4, 5.65), (13.4, 5.2), rad=0.0)

    # 7. Action sample
    draw_box(
        ax,
        x=7.4,
        y=4.0,
        w=2.9,
        h=1.2,
        title="Sample action a_t",
        text="train: ε-greedy over feasible\n"
             "       else  Categorical(π)\n"
             "eval : argmax(π)",
        color=COLORS["action"],
        fontsize=8.5,
        title_fontsize=10.0,
    )

    arrow(ax, (10.7, 4.6), (10.3, 4.6))

    # ------------------------------------------------------------------
    # Row 4: environment step
    # ------------------------------------------------------------------

    # 8. Environment / RDKit reaction
    draw_box(
        ax,
        x=3.6,
        y=4.0,
        w=3.4,
        h=1.2,
        title="Environment step (RDKit)",
        text="if a_t == Stop  → terminate\n"
             "else apply template a_t to s_t\n"
             "         → s_{t+1},  r_t = ΔQED",
        color=COLORS["env"],
        fontsize=8.5,
        title_fontsize=10.0,
    )

    arrow(ax, (7.4, 4.6), (7.0, 4.6))

    # 9. Loop back: s_{t+1} -> next iteration of the policy (back to box 1)
    arrow(
        ax,
        (3.6, 4.6),
        (1.3, 6.0),
        rad=-0.35,
        text="s_{t+1}",
        text_offset=(-0.4, 0.0),
        ls="--",
    )

    # ------------------------------------------------------------------
    # Row 5 (bottom): trajectory, replay buffer, TB loss
    # ------------------------------------------------------------------

    # 10. Trajectory
    draw_box(
        ax,
        x=0.2,
        y=1.6,
        w=3.6,
        h=1.4,
        title="Trajectory τ",
        text="(s_0, a_1, r_1, s_1, …, s_T)\n"
             "R(τ) = Σ_t r_t  =  QED(s_T) − QED(s_0)",
        color=COLORS["env"],
        fontsize=8.5,
        title_fontsize=10.5,
    )

    # 11. Best-K replay buffer
    draw_box(
        ax,
        x=4.2,
        y=1.6,
        w=3.4,
        h=1.4,
        title="Best-K replay buffer",
        text="top-K trajectories by R(τ)\n"
             "replayed each step with prob.\n"
             "replay_prob (deterministic re-eval)",
        color=COLORS["buffer"],
        fontsize=8.5,
        title_fontsize=10.5,
    )

    # 12. Trajectory Balance loss
    draw_box(
        ax,
        x=8.0,
        y=1.6,
        w=4.2,
        h=1.4,
        title="Trajectory Balance loss",
        text=r"L = ( log Z + Σ_t log π(a_t|s_t)" "\n"
             r"          − β · log(R(τ)+1) )²" "\n"
             r"+ 1e-3 · ( mean_t logsumexp logits )²",
        color=COLORS["loss"],
        fontsize=8.8,
        title_fontsize=10.5,
    )

    # 13. Optimizer / parameter update
    draw_box(
        ax,
        x=12.6,
        y=1.6,
        w=3.6,
        h=1.4,
        title="Optimizer (Adam, grad-clip 10.0)",
        text="∇ updates ALL trainable params:\n"
             "encoder + template_head + stop_head\n"
             "+ learnable scalar log Z",
        color=COLORS["loss"],
        fontsize=8.5,
        title_fontsize=10.5,
    )

    # arrows along bottom row.
    arrow(ax, (3.8, 2.3), (4.2, 2.3), text="push", text_offset=(0, 0.18), text_fontsize=8.0)
    arrow(ax, (7.6, 2.3), (8.0, 2.3), text="batch", text_offset=(0, 0.18), text_fontsize=8.0)
    arrow(ax, (12.2, 2.3), (12.6, 2.3))

    # trajectory finishes -> bottom row
    arrow(
        ax,
        (5.3, 4.0),
        (3.0, 3.0),
        rad=0.25,
        text="end of episode",
        text_offset=(-0.6, 0.05),
    )

    # Replay flag: the buffer -> loss arrow already represents replay; we just
    # add a small annotation under the buffer to make this explicit.
    ax.text(
        5.9,
        1.5,
        "(stored τ replays through the same TB loss with prob. replay_prob)",
        ha="center",
        va="top",
        fontsize=7.8,
        color="#5b21b6",
        style="italic",
        **FONT,
    )

    # ------------------------------------------------------------------
    # Legend.
    # ------------------------------------------------------------------

    legend_items = [
        ("Input / data",         COLORS["input"]),
        ("Graph encoder",        COLORS["encoder"]),
        ("Action heads",         COLORS["head"]),
        ("Action selection",     COLORS["action"]),
        ("Environment / τ",      COLORS["env"]),
        ("Replay buffer",        COLORS["buffer"]),
        ("Loss / optimizer",     COLORS["loss"]),
    ]
    lx = 0.2
    ly = 0.25
    for i, (label, color) in enumerate(legend_items):
        x = lx + i * 2.18
        rect = FancyBboxPatch(
            (x, ly),
            0.35,
            0.35,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=BOX_LW,
            edgecolor=COLORS["border"],
            facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(
            x + 0.45,
            ly + 0.18,
            label,
            ha="left",
            va="center",
            fontsize=9,
            color="#111827",
            **FONT,
        )

    return fig


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    png = out_dir / "graphtransrl_workflow.png"
    pdf = out_dir / "graphtransrl_workflow.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
