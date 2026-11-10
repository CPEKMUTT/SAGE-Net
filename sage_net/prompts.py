"""Structured prompts for the SSPB.

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

# UC-Merced folder names are written without separators.
_JOINED_NAMES = {
    "baseballdiamond": "baseball_diamond",
    "denseresidential": "dense_residential",
    "golfcourse": "golf_course",
    "mediumresidential": "medium_residential",
    "mobilehomepark": "mobile_home_park",
    "parkinglot": "parking_lot",
    "sparseresidential": "sparse_residential",
    "storagetanks": "storage_tanks",
    "tenniscourt": "tennis_court",
}


def normalize_class_name(name: str) -> str:
    """'DenseResidential', 'dense residential', 'denseresidential' -> 'dense_residential'."""
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name.strip())
    name = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return _JOINED_NAMES.get(name, name)


def readable(name: str) -> str:
    return normalize_class_name(name).replace("_", " ")


# class -> (objects, layout). The layout text completes "A <class> ...".
DESCRIPTIONS: dict[str, tuple[str, str]] = {
    # NWPU-RESISC45
    "airplane": (
        "aircraft, wings, fuselages, and airport pavement",
        "often shows aircraft with symmetric elongated bodies and swept wings parked on paved surfaces"),
    "airport": (
        "runways, taxiways, aircraft, terminals, and service roads",
        "often shows long parallel pavement strips connected to aprons and terminal buildings"),
    "baseball_diamond": (
        "an infield, bases, an outfield, and surrounding grass",
        "often shows a diamond-shaped infield at the corner of a fan-shaped field"),
    "basketball_court": (
        "court surfaces, painted lines, hoops, and nearby paths",
        "often shows small rectangular courts with symmetric painted markings"),
    "beach": (
        "sand, waves, sea water, and coastal vegetation",
        "often shows an elongated sandy strip running along the boundary between land and water"),
    "bridge": (
        "a road or railway deck, piers, water, and riverbanks",
        "often shows a long narrow linear structure spanning across water or a valley"),
    "chaparral": (
        "shrubs, bushes, dry soil, and sparse vegetation",
        "often shows irregular scattered patches of shrubs without a regular structure"),
    "church": (
        "a church building, roofs, towers, and courtyards",
        "often shows a prominent cross-shaped or elongated building surrounded by smaller urban blocks"),
    "circular_farmland": (
        "cultivated crop fields, center-pivot irrigation, and farm tracks",
        "often shows large circular fields arranged side by side in a regular pattern"),
    "cloud": (
        "bright cloud masses, thin haze, and shadows",
        "often shows large irregular bright regions with soft diffuse boundaries"),
    "commercial_area": (
        "shops, office buildings, roads, and parking lots",
        "often shows large building blocks interleaved with streets and parking areas"),
    "dense_residential": (
        "closely packed houses, rooftops, and narrow streets",
        "often shows small buildings densely arranged in compact rows with little open space"),
    "desert": (
        "sand, dunes, bare soil, and very little vegetation",
        "often shows wide homogeneous areas with wave-like dune patterns"),
    "forest": (
        "dense trees, canopy crowns, and vegetation",
        "often shows a continuous textured canopy covering the whole scene"),
    "freeway": (
        "multi-lane roads, vehicles, medians, and road shoulders",
        "often shows long parallel lanes running straight or in smooth curves across the scene"),
    "golf_course": (
        "fairways, greens, sand bunkers, trees, and ponds",
        "often shows winding elongated grass areas with scattered bunkers and tree lines"),
    "ground_track_field": (
        "a running track, a sports field, and stands",
        "often shows an oval running track enclosing a rectangular field"),
    "harbor": (
        "water, ships, docks, piers, and port infrastructure",
        "often contains ships and docks arranged along water boundaries, with linear piers connected to coastal areas"),
    "industrial_area": (
        "factories, warehouses, storage yards, and roads",
        "often shows large rectangular roofs separated by wide service roads"),
    "intersection": (
        "crossing roads, lane markings, vehicles, and traffic islands",
        "often shows two or more straight roads meeting at a central junction"),
    "island": (
        "land, vegetation, beaches, and surrounding water",
        "often shows a compact land region completely enclosed by water"),
    "lake": (
        "open inland water, shorelines, and surrounding land",
        "often shows a large smooth water body with an irregular closed shoreline"),
    "meadow": (
        "grass, low vegetation, and open land",
        "often shows broad uniform grass areas with very few structures"),
    "medium_residential": (
        "houses, gardens, trees, and streets",
        "often shows moderately spaced houses arranged along regular street grids"),
    "mobile_home_park": (
        "mobile homes, access roads, and small yards",
        "often shows many small identical rectangular homes aligned in parallel rows"),
    "mountain": (
        "ridges, slopes, rocks, and vegetation",
        "often shows rugged terrain with branching ridges and valleys"),
    "overpass": (
        "elevated roads, ramps, bridges, and crossing roads",
        "often shows roads crossing over each other with curved connecting ramps"),
    "palace": (
        "large historic buildings, courtyards, and gardens",
        "often shows a symmetric building complex organised around enclosed courtyards"),
    "parking_lot": (
        "parked cars, marked parking spaces, and pavement",
        "often shows vehicles arranged in dense regular rows on a paved area"),
    "railway": (
        "rail tracks, sleepers, ballast, and rail-side land",
        "often shows long parallel tracks forming narrow straight or gently curved corridors"),
    "railway_station": (
        "platforms, rail tracks, trains, and station buildings",
        "often shows many parallel tracks converging next to long platforms and a station building"),
    "rectangular_farmland": (
        "crop fields, field boundaries, and farm roads",
        "often shows adjacent rectangular plots separated by straight boundaries"),
    "river": (
        "flowing water, riverbanks, and riparian vegetation",
        "often shows a long winding water corridor crossing the scene"),
    "roundabout": (
        "a circular road, a central island, and connecting roads",
        "often shows a ring-shaped road with several roads radiating outward"),
    "runway": (
        "a paved runway, markings, and surrounding grass",
        "often shows a single very long straight paved strip with painted markings"),
    "sea_ice": (
        "floating ice sheets, cracks, and dark sea water",
        "often shows broken irregular ice plates separated by narrow channels of water"),
    "ship": (
        "vessels, wakes, and open water",
        "often shows one or a few elongated hulls surrounded by water"),
    "snowberg": (
        "snow, ice, exposed rocks, and mountain ridges",
        "often shows bright snow surfaces broken by dark ridges and slopes"),
    "sparse_residential": (
        "scattered houses, trees, lawns, and roads",
        "often shows isolated buildings separated by large areas of vegetation"),
    "stadium": (
        "a sports field, stands, roofs, and parking areas",
        "often shows a large oval or rectangular arena surrounded by seating"),
    "storage_tank": (
        "cylindrical tanks, pipelines, and industrial roads",
        "often shows groups of circular tank roofs arranged in regular clusters"),
    "tennis_court": (
        "tennis courts, nets, painted lines, and fences",
        "often shows small rectangular courts placed side by side"),
    "terrace": (
        "stepped fields, slopes, and embankments",
        "often shows narrow curved bands of fields following the terrain contours"),
    "thermal_power_station": (
        "cooling towers, chimneys, industrial buildings, and coal yards",
        "often shows large circular cooling towers next to rectangular industrial blocks"),
    "wetland": (
        "shallow water, marsh vegetation, and channels",
        "often shows irregular patches of water and vegetation mixed together"),
    # AID classes not covered above
    "bare_land": (
        "exposed soil, sand, and sparse ground cover",
        "often shows wide open land with few objects and little vegetation"),
    "center": (
        "a central landmark building, plazas, and roads",
        "often shows a large distinctive building surrounded by open space and road rings"),
    "farmland": (
        "crop fields, field boundaries, and farm roads",
        "often shows regular field parcels separated by straight boundaries"),
    "park": (
        "trees, lawns, footpaths, and ponds",
        "often shows green open areas crossed by curved paths within an urban surrounding"),
    "playground": (
        "a running track, a sports field, and courts",
        "often shows an oval track around a field next to smaller courts"),
    "pond": (
        "a small water body, banks, and nearby vegetation",
        "often shows compact enclosed water regions surrounded by land"),
    "resort": (
        "villas, swimming pools, gardens, and paths",
        "often shows low buildings scattered among landscaped green areas"),
    "school": (
        "school buildings, a playground, and sports fields",
        "often shows rectangular buildings arranged around a running track or yard"),
    "square": (
        "a paved plaza, lawns, paths, and surrounding buildings",
        "often shows a large open geometric space framed by roads and buildings"),
    "viaduct": (
        "elevated roads, supporting pillars, and ramps",
        "often shows long elevated corridors with interchanging curved ramps"),
    # UC-Merced classes not covered above
    "agricultural": (
        "crop fields, soil, and farm tracks",
        "often shows regularly textured fields with parallel row patterns"),
    "buildings": (
        "building roofs, streets, and paved surfaces",
        "often shows clusters of rectangular roofs separated by streets"),
}

# Dataset-specific names that share a description with another class.
_ALIASES = {
    "baseball_field": "baseball_diamond",
    "commercial": "commercial_area",
    "industrial": "industrial_area",
    "parking": "parking_lot",
    "port": "harbor",
    "storage_tanks": "storage_tank",
    "airplanes": "airplane",
}

PROMPT_SETS = {
    "classname": ("classname",),
    "identity": ("identity",),
    "identity_object": ("identity", "object"),
    "identity_geometry": ("identity", "geometry"),
    "full": ("identity", "object", "geometry"),
}


def _article(word: str) -> str:
    return "an" if word[0] in "aeiou" else "a"


def load_descriptions(overrides: str | Path | None = None) -> dict[str, tuple[str, str]]:
    table = dict(DESCRIPTIONS)
    if overrides:
        data = json.loads(Path(overrides).read_text(encoding="utf-8"))
        for name, entry in data.items():
            table[normalize_class_name(name)] = (entry["objects"], entry["layout"])
    return table


def class_prompts(name: str, descriptions: dict[str, tuple[str, str]]) -> dict[str, str]:
    key = normalize_class_name(name)
    text = key.replace("_", " ")
    lookup = key if key in descriptions else _ALIASES.get(key, key)
    if lookup not in descriptions:
        raise KeyError(f"no prompt description for class {name!r}; add it with --prompt-overrides")
    objects, layout = descriptions[lookup]
    return {
        "classname": text,
        "identity": f"A remote sensing image of {_article(text)} {text}.",
        "object": f"{_article(text).capitalize()} {text} scene usually contains {objects}.",
        "geometry": f"{_article(text).capitalize()} {text} {layout}.",
    }


def build_prompts(class_names: Sequence[str], prompt_set: str = "full",
                  overrides: str | Path | None = None) -> list[list[str]]:
    """Prompt texts per class, ordered as in PROMPT_SETS[prompt_set]."""
    kinds = PROMPT_SETS[prompt_set]
    descriptions = load_descriptions(overrides)
    return [[class_prompts(c, descriptions)[k] for k in kinds] for c in class_names]


@torch.no_grad()
def encode_prompts(encoder, class_names: Sequence[str], prompt_set: str = "full",
                   overrides: str | Path | None = None) -> Tensor:
    """Encode the prompt bank once. Returns [C, M, d] with M prompts per class."""
    prompts = build_prompts(class_names, prompt_set, overrides)
    flat = [p for per_class in prompts for p in per_class]
    emb = encoder.encode_text(flat)
    return emb.view(len(class_names), len(prompts[0]), -1)
