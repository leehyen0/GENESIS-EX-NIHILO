from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, Mapping, Optional, Protocol, Sequence, Tuple
import hashlib
import hmac
import json

from .world_coupling import WorldOutcomePair


@dataclass(frozen=True)
class RawObservationReceipt:
    """Externally signed raw observation bound to one executed world challenge.

    Raw channels are representation inputs, not authority by themselves. A BODY may
    use their numeric values for primitive genesis only after an external verifier
    authenticates the exact channel payload and the receipt is bound to an already
    authoritative WorldOutcomePair from the same issuer/source/context/challenge.
    """

    observation_id: str
    intervention_id: str
    channel_values: Tuple[Tuple[str, float], ...]
    source_id: str
    context_id: str
    challenge_id: str
    epoch: int
    externally_generated: bool = True
    issuer_id: str = "UNSIGNED"
    signature: str = ""

    @property
    def normalized_values(self) -> Tuple[Tuple[str, float], ...]:
        return tuple(sorted((str(name), float(value)) for name, value in self.channel_values))


def raw_observation_payload(receipt: RawObservationReceipt) -> bytes:
    data = {
        "observation_id": receipt.observation_id,
        "intervention_id": receipt.intervention_id,
        "channel_values": [[name, float(value)] for name, value in receipt.normalized_values],
        "source_id": receipt.source_id,
        "context_id": receipt.context_id,
        "challenge_id": receipt.challenge_id,
        "epoch": int(receipt.epoch),
        "externally_generated": bool(receipt.externally_generated),
        "issuer_id": receipt.issuer_id,
    }
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


class RawObservationVerifier(Protocol):
    def verify(self, receipt: RawObservationReceipt) -> bool:
        ...

    def independence_class(self, receipt: RawObservationReceipt) -> str:
        ...


class HMACRawObservationSigner:
    """Source-side reference signer. Secret material is never BODY state."""

    def __init__(self, issuer_id: str, secret: bytes) -> None:
        if not issuer_id:
            raise ValueError("issuer_id is required")
        if not secret:
            raise ValueError("signing secret is required")
        self.issuer_id = str(issuer_id)
        self._secret = bytes(secret)

    def sign(self, receipt: RawObservationReceipt) -> RawObservationReceipt:
        unsigned = replace(receipt, issuer_id=self.issuer_id, signature="")
        signature = hmac.new(self._secret, raw_observation_payload(unsigned), hashlib.sha256).hexdigest()
        return replace(unsigned, signature=signature)


class HMACRawObservationVerifier:
    """LAB-side authenticity and independence authority for raw observations."""

    def __init__(
        self,
        trusted_keys: Mapping[str, bytes],
        independence_classes: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._trusted_keys = {
            str(issuer): bytes(secret)
            for issuer, secret in trusted_keys.items()
            if issuer and secret
        }
        supplied = dict(independence_classes or {})
        self._independence_classes = {
            issuer: str(supplied.get(issuer, issuer))
            for issuer in self._trusted_keys
            if str(supplied.get(issuer, issuer))
        }

    def verify(self, receipt: RawObservationReceipt) -> bool:
        secret = self._trusted_keys.get(receipt.issuer_id)
        if secret is None or not receipt.signature:
            return False
        expected = hmac.new(
            secret,
            raw_observation_payload(replace(receipt, signature="")),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, receipt.signature)

    def independence_class(self, receipt: RawObservationReceipt) -> str:
        if receipt.issuer_id not in self._trusted_keys:
            return "UNVERIFIED"
        return self._independence_classes.get(receipt.issuer_id, receipt.issuer_id)


def _world_pair_matches(receipt: RawObservationReceipt, pair: WorldOutcomePair) -> bool:
    return bool(
        pair.authority_verified
        and pair.matched_budget
        and pair.externally_generated
        and pair.independence_class_id != "UNVERIFIED"
        and pair.experiment_id == receipt.intervention_id
        and pair.issuer_id == receipt.issuer_id
        and pair.source_id == receipt.source_id
        and pair.context_id == receipt.context_id
        and pair.challenge_id == receipt.challenge_id
        and int(pair.epoch) == int(receipt.epoch)
        and bool(receipt.externally_generated)
    )


def corroborated_raw_observations(
    receipts: Iterable[RawObservationReceipt],
    world_pairs: Sequence[WorldOutcomePair],
    verifier: Optional[RawObservationVerifier],
    min_independent_classes: int = 2,
) -> Dict[str, Dict[str, float]]:
    """Return only raw rows independently corroborated and world-pair bound.

    For each intervention, receipts are first authenticated, then matched to an
    authoritative WorldOutcomePair with the same execution identity. Distinct
    verifier-derived independence classes must agree on the exact normalized raw
    channel map. Conflicting quorums fail closed rather than choosing one row.
    """

    if verifier is None:
        return {}
    minimum = max(1, int(min_independent_classes))
    pair_by_identity = {
        (
            pair.experiment_id,
            pair.issuer_id,
            pair.source_id,
            pair.context_id,
            pair.challenge_id,
            int(pair.epoch),
        ): pair
        for pair in world_pairs
    }
    grouped: Dict[str, Dict[Tuple[Tuple[str, float], ...], set[str]]] = {}
    for receipt in receipts:
        if not verifier.verify(receipt):
            continue
        independence = str(verifier.independence_class(receipt) or "UNVERIFIED")
        if independence == "UNVERIFIED":
            continue
        identity = (
            receipt.intervention_id,
            receipt.issuer_id,
            receipt.source_id,
            receipt.context_id,
            receipt.challenge_id,
            int(receipt.epoch),
        )
        pair = pair_by_identity.get(identity)
        if pair is None or not _world_pair_matches(receipt, pair):
            continue
        if pair.independence_class_id != independence:
            continue
        values = receipt.normalized_values
        if not values:
            continue
        grouped.setdefault(receipt.intervention_id, {}).setdefault(values, set()).add(independence)

    result: Dict[str, Dict[str, float]] = {}
    for intervention_id, signatures in grouped.items():
        qualified = [
            values for values, classes in signatures.items()
            if len(classes) >= minimum
        ]
        if len(qualified) != 1:
            continue
        result[intervention_id] = {name: float(value) for name, value in qualified[0]}
    return result
