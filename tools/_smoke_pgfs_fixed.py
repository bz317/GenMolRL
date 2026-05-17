"""Quick smoke test for Bi-TD3 PGFS-Fixed code changes.

Three Bi-TD3-only fixes to verify:
  (A) ``td3/knn.py``: drop binariser; rescale FAISS keys → {-1, +1}.
  (B) ``td3/train.py::_to_r2_tensor``: warm-up R2 storage scaled → {-1, +1}.
  (C) ``td3/models.py::PiNetwork``: trailing 167-dim bottleneck removed.

Unit-level checks (no slow Bi env init):
  - PiNetwork inferred shape matches new defaults.
  - ``_to_r2_tensor`` on a SMILES path rescales {0, 1} → {-1, +1}.
  - ``_to_r2_tensor`` on a torch.Tensor short-circuits unchanged.
  - ``_to_r2_tensor`` on uni-discrete env returns shape (1, 0).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from genmolrl.algorithms.td3.constants import TD3_UNI_DISCRETE_ACTION_DESIGN  # noqa: E402
from genmolrl.algorithms.td3.models import PiNetwork, ActorNetwork  # noqa: E402
from genmolrl.algorithms.td3.train import _to_r2_tensor  # noqa: E402


def test_pinetwork_default_no_bottleneck() -> None:
    """Final hidden layer should now be 256 (was 167 pre-fix)."""
    net = PiNetwork(input_dim=1024 + 88, output_dim=1024)
    last_linear = [m for m in net.network if isinstance(m, torch.nn.Linear)][-1]
    assert last_linear.in_features == 256, last_linear
    assert last_linear.out_features == 1024
    print(
        f"  PiNetwork final Linear: in_features={last_linear.in_features} "
        f"(was 167 pre-fix), out_features={last_linear.out_features}"
    )


def test_actornetwork_uses_new_pinet() -> None:
    """ActorNetwork's pi_net must also pick up the new default."""
    actor = ActorNetwork(state_dim=1024, template_dim=88, action_dim=1024)
    pi_last = [m for m in actor.pi_net.network if isinstance(m, torch.nn.Linear)][-1]
    assert pi_last.in_features == 256
    print(
        f"  ActorNetwork.pi_net final Linear in_features={pi_last.in_features}"
    )


class _FakeBiEnv:
    """Minimal Bi env stub for ``_to_r2_tensor`` SMILES branch."""

    def __init__(self, fp_dim: int = 8):
        self.action_design = "pgfs_continuous_r2"
        self.base_obs_dim = fp_dim
        smiles = "CCO"
        bits = np.array([0, 1, 0, 1, 1, 0, 0, 1], dtype=np.float32)
        self.reactants = {smiles: bits}
        self.observation_space = SimpleNamespace(shape=(fp_dim,))


class _FakeUniDiscreteEnv:
    """Minimal uni-discrete stub: must return shape (1, 0) regardless of input."""

    def __init__(self):
        self.action_design = TD3_UNI_DISCRETE_ACTION_DESIGN


def test_to_r2_tensor_rescales_smiles() -> None:
    env = SimpleNamespace(unwrapped=_FakeBiEnv())
    out = _to_r2_tensor(env, "CCO")
    expected = torch.tensor([[-1.0, +1.0, -1.0, +1.0, +1.0, -1.0, -1.0, +1.0]])
    assert out.shape == expected.shape
    assert torch.allclose(out.cpu(), expected), (out, expected)
    print(f"  SMILES warm-up → r2 vec rescaled to {out.tolist()[0]}")


def test_to_r2_tensor_passes_tensor_through() -> None:
    env = SimpleNamespace(unwrapped=_FakeBiEnv())
    tanh_like = torch.tensor([[+0.7, -0.4, +0.9, -0.95, +0.05, -0.6, +0.8, -0.3]])
    out = _to_r2_tensor(env, tanh_like)
    assert out is tanh_like
    print("  Tensor R2 (actor output) is returned unchanged")


def test_to_r2_tensor_uni_returns_empty() -> None:
    env = SimpleNamespace(unwrapped=_FakeUniDiscreteEnv())
    out = _to_r2_tensor(env, "anything")
    assert out.shape == (1, 0), out.shape
    print(f"  Uni-discrete → empty R2 (shape={tuple(out.shape)})")


def test_to_r2_tensor_none_returns_zeros() -> None:
    env = SimpleNamespace(unwrapped=_FakeBiEnv())
    out = _to_r2_tensor(env, None)
    assert out.shape == (1, 8)
    assert torch.all(out == 0.0)
    print(f"  None R2 → zeros vector (shape={tuple(out.shape)})")


def test_knn_index_scaled_to_pm1() -> None:
    """KNN FAISS keys should now be in {-1, +1}, not {0, 1}."""
    from genmolrl.algorithms.td3.knn import KNNWrapper  # noqa: WPS433

    # Build a fake wrapper instance bypassing __init__ via __new__
    wrapper = KNNWrapper.__new__(KNNWrapper)
    wrapper.knn_indices = {}
    wrapper.use_gpu = False
    wrapper.res = None
    wrapper.top_k = 5
    wrapper.score_mode = "product"
    wrapper.random_epsilon = 0.0

    # Synthetic env exposing one bimolecular template + 3 reactants
    fp_dim = 4
    reactants = {
        "A": np.array([1, 0, 1, 0], dtype=np.float32),
        "B": np.array([0, 0, 1, 1], dtype=np.float32),
        "C": np.array([1, 1, 0, 0], dtype=np.float32),
    }

    class _RM:
        def get_valid_reactants(self, _idx):
            return list(reactants.keys())

    class _Env:
        action_design = "pgfs_continuous_r2"
        templates = [{"type": "bimolecular"}]
        reaction_manager = _RM()

    wrapper.env = SimpleNamespace(unwrapped=_Env())
    wrapper.reactants = reactants
    wrapper._initialize_index_for_template(0)

    index = wrapper.knn_indices[0]
    # FAISS IndexFlatL2 exposes ``reconstruct`` for each vector.
    reconstructed = np.stack([index.reconstruct(i) for i in range(index.ntotal)])
    print(f"  FAISS key matrix (rescaled to ±1):\n{reconstructed}")
    assert set(np.unique(reconstructed)).issubset({-1.0, 1.0}), reconstructed


if __name__ == "__main__":
    print("[A] PiNetwork bottleneck removed")
    test_pinetwork_default_no_bottleneck()
    test_actornetwork_uses_new_pinet()

    print("\n[B] _to_r2_tensor rescales warm-up storage")
    test_to_r2_tensor_rescales_smiles()
    test_to_r2_tensor_passes_tensor_through()
    test_to_r2_tensor_uni_returns_empty()
    test_to_r2_tensor_none_returns_zeros()

    print("\n[C] KNN FAISS index keys rescaled to ±1")
    test_knn_index_scaled_to_pm1()

    print("\nSMOKE OK")
