from __future__ import annotations

from typing import Mapping, Optional

from .software_repair_constructor_genesis import (
    CONSTRUCTOR_FAMILY,
    ConstructorInexpressivityAssessment,
    RelationalConstructorPrimitive,
    RelationalRepairConstructorPolicy,
    infer_relational_constructor_primitive,
)


def infer_descendant_relational_constructor_primitive(
    stderr: str,
    source: str,
    target_path: str,
    repository_sources: Mapping[str, str],
    assessment: ConstructorInexpressivityAssessment,
    policy: RelationalRepairConstructorPolicy,
) -> Optional[RelationalConstructorPrimitive]:
    """Expand the constructor only after its meta-policy is externally revalidated.

    Training-time primitive proposals may exist in shadow. A descendant may use the
    relational constructor on a previously unseen exception relation only when the
    constructor family has repeated, verifier-derived world support. This keeps
    phenotype heredity separate from epistemic/action authority.
    """
    if (
        policy.status != "REPRODUCED_RELATIONAL_REPAIR_CONSTRUCTOR"
        or policy.constructor_family != CONSTRUCTOR_FAMILY
    ):
        return None
    return infer_relational_constructor_primitive(
        stderr=stderr,
        source=source,
        target_path=target_path,
        repository_sources=repository_sources,
        assessment=assessment,
    )
