"""Product selection policies for RDKit reaction outcomes."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import QED


def best_qed_product_smiles(product_sets) -> str | None:
    """Select the first product from each product set, sanitized, maximizing QED."""
    valid_products = []
    for product_set in product_sets or []:
        if not product_set:
            continue
        product = product_set[0]
        try:
            if Chem.SanitizeMol(product, catchErrors=True) == Chem.SanitizeFlags.SANITIZE_NONE:
                valid_products.append(product)
        except Exception:
            continue
    if not valid_products:
        return None
    try:
        return Chem.MolToSmiles(max(valid_products, key=QED.qed))
    except Exception:
        return None
