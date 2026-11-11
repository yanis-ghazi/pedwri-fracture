import re

AO_RE = re.compile(r"(?P<loc>\d{2})(?P<bone>[ru]?)-(?P<segment>[EMD])/(?P<type>\d)(?:\.(?P<sub>\d+))?")
BONES = {"r": "radius", "u": "ulna", "": "radius+ulna"}


def parse(codes: str) -> list[dict]:
    return [m.groupdict() for m in AO_RE.finditer(codes or "")]


def pattern(segment: str, type_: str) -> str:
    if segment == "M":
        return {"2": "metaphyseal incomplete", "3": "metaphyseal complete"}.get(type_, "metaphyseal other")
    if segment == "E":
        if type_ in {"1", "2", "3", "4"}:
            return "physeal (Salter-Harris)"
        if type_ == "7":
            return "avulsion"
        return "epiphyseal other"
    return "diaphyseal"


def patterns(codes: str) -> set[str]:
    return {pattern(c["segment"], c["type"]) for c in parse(codes)}


def detailed(codes: str) -> set[str]:
    return {f"{BONES[c['bone']]} {c['segment']}/{c['type']}" for c in parse(codes)}
