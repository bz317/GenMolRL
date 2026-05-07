"""TD3 template-mask selection for policy logits, replay batches, and warmup.

TD3 uses the same ``masking`` field as the rest of GenMolRL (``none``, ``substructure``,
``reaction_valid``, ``r2_available``) via ``env.mask_provider.mode``. Stop is appended
separately when enabled.

``reaction_mode: bi`` means the **full template pool** (unimolecular + bimolecular
templates together); there is no separate "purely bi-only" mode in this codebase.

Set ``td3.template_mask_kind`` to override the env masking kind for TD3-only (e.g.
experiments where TD3 should see a different feasibility notion than SB3 on the same env).
"""

from __future__ import annotations


def td3_template_mask_kind(env, *, override: str | None = None) -> str:
    if override is not None:
        return override
    return getattr(env.unwrapped.mask_provider, "mode", "r2_available")
