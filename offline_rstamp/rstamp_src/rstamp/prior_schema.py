"""Structured reasoning prior schema for R-STAMP.

The prior is deliberately short. It is not a long chain-of-thought; it is a
compact interface between language reasoning and parallel mask prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable


Point = tuple[float, float]
BBox = tuple[float, float, float, float]


@dataclass
class StructuredPrior:
    target: str = ""
    attributes: list[str] = field(default_factory=list)
    relation: str = ""
    bbox: BBox | None = None
    positive_points: list[Point] = field(default_factory=list)
    negative_points: list[Point] = field(default_factory=list)
    ambiguity: str = ""


def _fmt_points(points: Iterable[Point]) -> str:
    return ";".join(f"{x:.1f},{y:.1f}" for x, y in points)


def format_prior(prior: StructuredPrior) -> str:
    """Format a prior as compact tagged text."""
    parts = []
    if prior.target:
        parts.append(f"<TARGET> {prior.target} </TARGET>")
    if prior.attributes:
        parts.append(f"<ATTR> {', '.join(prior.attributes)} </ATTR>")
    if prior.relation:
        parts.append(f"<REL> {prior.relation} </REL>")
    if prior.bbox is not None:
        x1, y1, x2, y2 = prior.bbox
        parts.append(f"<BOX> {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} </BOX>")
    if prior.positive_points:
        parts.append(f"<POS> {_fmt_points(prior.positive_points)} </POS>")
    if prior.negative_points:
        parts.append(f"<NEG> {_fmt_points(prior.negative_points)} </NEG>")
    if prior.ambiguity:
        parts.append(f"<AMB> {prior.ambiguity} </AMB>")
    return " ".join(parts)


def _tag(text: str, name: str) -> str:
    match = re.search(rf"<{name}>\s*(.*?)\s*</{name}>", text, flags=re.I | re.S)
    return match.group(1).strip() if match else ""


def _parse_points(value: str) -> list[Point]:
    points: list[Point] = []
    for item in value.split(";"):
        if not item.strip():
            continue
        nums = re.findall(r"-?\d+(?:\.\d+)?", item)
        if len(nums) >= 2:
            points.append((float(nums[0]), float(nums[1])))
    return points


def parse_prior_text(text: str) -> StructuredPrior:
    """Parse compact tagged prior text back into a dataclass."""
    attrs = [x.strip() for x in _tag(text, "ATTR").split(",") if x.strip()]
    box_nums = re.findall(r"-?\d+(?:\.\d+)?", _tag(text, "BOX"))
    bbox = tuple(float(x) for x in box_nums[:4]) if len(box_nums) >= 4 else None
    return StructuredPrior(
        target=_tag(text, "TARGET"),
        attributes=attrs,
        relation=_tag(text, "REL"),
        bbox=bbox,  # type: ignore[arg-type]
        positive_points=_parse_points(_tag(text, "POS")),
        negative_points=_parse_points(_tag(text, "NEG")),
        ambiguity=_tag(text, "AMB"),
    )


def prior_is_valid(text: str) -> bool:
    prior = parse_prior_text(text)
    has_target = bool(prior.target)
    has_location = prior.bbox is not None or bool(prior.positive_points)
    return has_target and has_location

