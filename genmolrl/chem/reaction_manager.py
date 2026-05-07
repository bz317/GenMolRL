"""Shared RDKit reaction manager for GenMolRL."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from genmolrl.chem.product_selection import best_qed_product_smiles

logger = logging.getLogger(__name__)

UNI_TYPES = {"unimolecular", "unimolecular_explicit_reagent"}
BI_TYPE = "bimolecular"


class ReactionManager:
    """Applies templates and provides masks for template/R2 selection."""

    def __init__(self, templates: dict, reactants: dict):
        self.templates = self._normalize_templates(templates)
        self.reactants = reactants
        self.template_mask_cache: dict[tuple[str | None, str], torch.Tensor] = {}
        self.valid_reactants_cache: dict[int, list[str]] = {}
        self.template_types = self._template_types_tensor()
        self.template_keys = list(self.templates.keys())

    @staticmethod
    def _normalize_templates(templates: dict) -> dict[int, dict[str, Any]]:
        out: dict[int, dict[str, Any]] = {}
        # Preserve pickle insertion order so action indices match the legacy PPO/A2C/TD3 code.
        for idx, (_, template) in enumerate(templates.items()):
            if not isinstance(template, dict):
                raise ValueError(f"Template {idx} must be a dict.")
            out[idx] = dict(template)
        return out

    def templates_for_mode(self, reaction_mode: str) -> dict[int, dict[str, Any]]:
        """``uni``: only UNI_TYPES templates. ``bi``: full pool (uni + bi templates together)."""
        if reaction_mode == "bi":
            return dict(self.templates)
        if reaction_mode != "uni":
            raise ValueError(f"Unsupported reaction_mode: {reaction_mode}")
        selected = [t for t in self.templates.values() if t.get("type") in UNI_TYPES]
        if not selected:
            raise ValueError("Uni mode requested but no unimolecular templates were found.")
        return {i: dict(t) for i, t in enumerate(selected)}

    def _template_types_tensor(self) -> torch.Tensor:
        mapping = {"unimolecular": 0, "unimolecular_explicit_reagent": 0, "bimolecular": 1}
        return torch.tensor(
            [mapping.get(t.get("type", "unimolecular"), 0) for t in self.templates.values()],
            dtype=torch.long,
        )

    @staticmethod
    def _template_smarts(template: dict | str) -> str:
        return template["smarts"] if isinstance(template, dict) else str(template)

    @staticmethod
    def _template_fixed_reagents(template: dict | str) -> list[str]:
        if isinstance(template, dict):
            return list(template.get("_explicit_reagents", []))
        return []

    @staticmethod
    def _mol_from_reagent_smarts(smarts: str):
        mol = Chem.MolFromSmiles(smarts)
        if mol is not None:
            return mol
        return Chem.MolFromSmarts(smarts)

    def apply_reaction(self, state: str | None, template: dict | str, reactant: str | None = None) -> str | None:
        if not state:
            return None
        try:
            state_mol = Chem.MolFromSmiles(state)
            if state_mol is None:
                return None
            reaction = AllChem.ReactionFromSmarts(self._template_smarts(template))
            product_sets = self._run_reaction(
                state_mol,
                reaction,
                reactant,
                self._template_fixed_reagents(template),
            )
            return best_qed_product_smiles(product_sets)
        except Exception as exc:
            logger.debug("Reaction failed for %s: %s", state, exc)
            return None

    def _run_reaction(self, state_mol, reaction, reactant: str | None, fixed_reagents: Iterable[str]):
        num_reactants = reaction.GetNumReactantTemplates()
        fixed = list(fixed_reagents or [])
        if fixed:
            reagent_mols = [self._mol_from_reagent_smarts(s) for s in fixed]
            if any(m is None for m in reagent_mols):
                return []
            if len(reagent_mols) == num_reactants - 1:
                return reaction.RunReactants((state_mol, *reagent_mols))
            return []
        if num_reactants == 1:
            return reaction.RunReactants((state_mol,))
        if num_reactants == 2 and reactant:
            reactant_mol = Chem.MolFromSmiles(reactant)
            if reactant_mol is not None:
                return reaction.RunReactants((state_mol, reactant_mol))
        return []

    def match_template(self, reactant: str | None, template: dict | str) -> dict[str, bool]:
        try:
            if not reactant:
                return {"first": False, "second": False}
            reaction = AllChem.ReactionFromSmarts(self._template_smarts(template))
            mol = Chem.MolFromSmiles(reactant)
            if mol is None:
                return {"first": False, "second": False}
            matches = {"first": False, "second": False}
            matches["first"] = mol.HasSubstructMatch(reaction.GetReactantTemplate(0), useChirality=True)
            if reaction.GetNumReactantTemplates() == 2:
                matches["second"] = mol.HasSubstructMatch(reaction.GetReactantTemplate(1), useChirality=True)
            return matches
        except Exception:
            return {"first": False, "second": False}

    def template_substructure_mask(self, reactant: str | None) -> torch.Tensor:
        """Validate templates only by RDKit first-reactant substructure match."""
        key = (reactant, "substructure")
        if key not in self.template_mask_cache:
            values = [int(self.match_template(reactant, t)["first"]) for t in self.templates.values()]
            self.template_mask_cache[key] = torch.tensor(values, dtype=torch.float32)
        return self.template_mask_cache[key].clone()

    def template_r2_available_mask(self, reactant: str | None) -> torch.Tensor:
        """Validate R1 match and, for bimolecular templates, availability of any R2.

        Unimolecular templates do not run RDKit product generation here; they only need
        a first-reactant substructure match. Bimolecular templates additionally need at
        least one pool molecule that matches the template's second reactant pattern.
        """
        key = (reactant, "r2_available")
        if key not in self.template_mask_cache:
            out = torch.zeros(len(self.templates), dtype=torch.float32)
            for idx, template in self.templates.items():
                if not self.match_template(reactant, template)["first"]:
                    continue
                ttype = template.get("type", "unimolecular")
                if ttype in UNI_TYPES:
                    out[idx] = 1.0
                elif ttype == BI_TYPE and self.get_valid_reactants(idx):
                    out[idx] = 1.0
            self.template_mask_cache[key] = out
        return self.template_mask_cache[key].clone()

    def template_reaction_valid_mask(self, reactant: str | None) -> torch.Tensor:
        """Validate by R1 match plus successful RDKit product generation.

        This is the exact legacy experiments-branch PPO/A2C mask for Uni runs:
        a template is valid iff the current molecule matches as first reactant and
        `apply_reaction(reactant, template, None)` returns a sanitized product.
        For true bimolecular templates this returns 0 unless a fixed reagent is encoded
        in the template, because no learned/selected R2 is supplied at mask time.
        """
        key = (reactant, "reaction_valid")
        if key not in self.template_mask_cache:
            values = []
            for template in self.templates.values():
                if self.match_template(reactant, template)["first"]:
                    values.append(1 if self.apply_reaction(reactant, template, None) else 0)
                else:
                    values.append(0)
            self.template_mask_cache[key] = torch.tensor(values, dtype=torch.float32)
        return self.template_mask_cache[key].clone()

    def get_mask(self, reactant: str | None, *, kind: str = "substructure") -> torch.Tensor:
        aliases = {
            "current": "substructure",
            "executable": "r2_available",
            "ppo_original": "reaction_valid",
        }
        kind = aliases.get(kind, kind)
        if kind == "none":
            return torch.ones(len(self.templates), dtype=torch.float32)
        if kind == "reaction_valid":
            return self.template_reaction_valid_mask(reactant)
        if kind == "r2_available":
            return self.template_r2_available_mask(reactant)
        if kind != "substructure":
            raise ValueError(f"Unsupported mask kind: {kind}")
        return self.template_substructure_mask(reactant)

    def get_feasible_mask(self, reactant: str | None) -> torch.Tensor:
        """Compatibility alias used by the existing PGFS TD3 agent."""
        return self.template_r2_available_mask(reactant)

    def feasible_first_reactant_templates(self, reactant: str | None, *, kind: str = "substructure") -> list[int]:
        mask = self.get_mask(reactant, kind=kind)
        return [int(i) for i in torch.where(mask > 0.5)[0]]

    def get_valid_reactants(self, template_index: int) -> list[str]:
        if template_index not in self.valid_reactants_cache:
            template = self.templates[int(template_index)]
            self.valid_reactants_cache[template_index] = [
                smiles for smiles in self.reactants if self.match_template(smiles, template)["second"]
            ]
        return list(self.valid_reactants_cache[template_index])

    def r2_mask(self, template_index: int) -> np.ndarray:
        mask = np.zeros(len(self.reactants), dtype=np.int8)
        valid = set(self.get_valid_reactants(template_index))
        for i, smiles in enumerate(self.reactants):
            if smiles in valid:
                mask[i] = 1
        return mask
