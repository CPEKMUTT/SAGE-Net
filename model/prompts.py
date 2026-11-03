from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor


def canonical_class_name(name: str) -> str:
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    aliases = {
        "port": "harbor", "commercial": "commercial_area",
        "industrial": "industrial_area", "parking": "parking_lot",
        "baseballfield": "baseball_diamond", "baseball_field": "baseball_diamond",
        "baseballdiamond": "baseball_diamond", "bareland": "bare_land",
        "bare_land": "bare_land", "center": "center",
        "farmland": "farmland", "playground": "playground",
        "railwaystation": "railway_station", "storage_tanks": "storage_tank",
        "storagetanks": "storage_tank", "storagetank": "storage_tank",
        "denseresidential": "dense_residential",
        "mediumresidential": "medium_residential",
        "sparseresidential": "sparse_residential",
        "mobilehomepark": "mobile_home_park",
        "tenniscourt": "tennis_court", "golfcourse": "golf_course",
        "parkinglot": "parking_lot", "groundtrackfield": "ground_track_field",
        "circularfarmland": "circular_farmland",
        "rectangularfarmland": "rectangular_farmland",
        "thermalpowerstation": "thermal_power_station",
        "snowberg": "snowberg",
    }
    return aliases.get(key, key)


# (object-composition description, geometry-layout description)
ATTRIBUTES: dict[str, tuple[str, str]] = {
    "airplane": ("aircraft and airport pavement", "aircraft bodies and wings forming symmetric elongated silhouettes"),
    "airport": ("runways, taxiways, aircraft, terminals, and service roads", "wide parallel pavement strips linked to apron and terminal regions"),
    "baseball_diamond": ("baseball infield, outfield, and surrounding facilities", "a diamond-shaped infield centered within a rounded field"),
    "basketball_court": ("court markings, basketball hoops, and paved play surfaces", "rectangular courts with symmetric painted lines"),
    "beach": ("sand, waves, water, and shore vegetation", "an elongated boundary between open water and sandy land"),
    "bridge": ("road or rail deck, supports, and a crossing corridor", "a long linear structure spanning water or lower terrain"),
    "chaparral": ("scrubland, bushes, and scattered dry vegetation", "irregular densely textured patches without a regular street grid"),
    "church": ("religious buildings, roofs, towers, and courtyards", "a prominent building footprint surrounded by smaller urban features"),
    "circular_farmland": ("cultivated fields, irrigation patterns, and farm tracks", "large circular plots with regular boundaries"),
    "cloud": ("bright cloud masses, shadowed edges, and sky-like textures", "irregular diffuse patches with soft boundaries"),
    "commercial_area": ("shops, offices, roads, and parking facilities", "large building footprints interspersed with parking and road grids"),
    "dense_residential": ("tightly packed houses, streets, and rooftops", "closely spaced repeated buildings and narrow road corridors"),
    "desert": ("bare soil, sand, dunes, and sparse vegetation", "broad open surfaces with low structural density"),
    "forest": ("trees, canopy crowns, and vegetation", "continuous irregular canopy texture and clustered tree crowns"),
    "freeway": ("high-speed road lanes, medians, and ramps", "long parallel road lanes and smooth sweeping connections"),
    "golf_course": ("fairways, greens, sand bunkers, and water hazards", "interconnected elongated green patches with curved edges"),
    "ground_track_field": ("running track, field, and spectator facilities", "oval lanes enclosing a rectangular central sports field"),
    "harbor": ("water, ships, docks, piers, and port infrastructure", "ships and docks along water boundaries, with linear piers connected to coastal areas"),
    "industrial_area": ("factories, warehouses, chimneys, roads, and storage yards", "large rectangular roofs and spaced service corridors"),
    "intersection": ("crossing roadways, lane markings, and traffic islands", "two or more road corridors meeting at a central junction"),
    "island": ("landmass, vegetation, and surrounding water", "a bounded land region enclosed by a water boundary"),
    "lake": ("inland water, banks, and surrounding terrain", "a broad enclosed water region with an irregular shoreline"),
    "meadow": ("grassland, low vegetation, and open terrain", "broad smooth vegetation patches with few built structures"),
    "medium_residential": ("houses, gardens, roads, and trees", "moderately spaced building footprints separated by streets and greenery"),
    "mobile_home_park": ("mobile homes, access roads, and shared open areas", "uniform small rectangular homes arranged in repeated rows"),
    "mountain": ("steep terrain, ridges, slopes, and vegetation", "large elevated ridges with oriented slopes and valleys"),
    "overpass": ("elevated roadway, ramps, bridges, and lower-level roads", "intersecting linear roads at distinct elevations"),
    "palace": ("large formal buildings, courtyards, and landscaped grounds", "symmetric building complexes and structured open courtyards"),
    "parking_lot": ("parked cars, marked stalls, and pavement", "repeating small rectangular vehicles in regular rows"),
    "railway": ("rail tracks, ballast, and rail-side facilities", "long straight parallel tracks forming continuous corridors"),
    "railway_station": ("rail platforms, tracks, shelters, and station buildings", "multiple parallel tracks adjacent to platforms and larger buildings"),
    "rectangular_farmland": ("cropland, field boundaries, and agricultural tracks", "adjacent rectangular plots with straight boundaries"),
    "river": ("flowing water, riverbanks, and riparian vegetation", "a continuous winding water corridor through surrounding land"),
    "roundabout": ("road lanes, traffic islands, and entering roads", "a circular central island connected to radial road branches"),
    "runway": ("paved runway, markings, and adjacent clear zones", "one or more very long straight pavement strips"),
    "sea_ice": ("sea ice, cracks, and dark ocean water", "irregular broken plates and branching fissures over water"),
    "ship": ("vessels, wakes, and surrounding water", "elongated compact hulls embedded in broad water regions"),
    "snowberg": ("snow-covered ridges, ice, and exposed rocks", "high-contrast elongated mountain slopes and ridges"),
    "sparse_residential": ("widely spaced houses, vegetation, and roads", "scattered buildings with large inter-building gaps"),
    "stadium": ("sports field, stands, seating, and access roads", "large oval or rectangular enclosed sports ground"),
    "storage_tank": ("cylindrical tanks, industrial pipes, and roads", "repeated circular tank footprints arranged in regular clusters"),
    "tennis_court": ("tennis courts, nets, and painted boundaries", "small repeated rectangles with symmetric center lines"),
    "terrace": ("stepped farm plots, slopes, and earth embankments", "stacked curving agricultural bands following contour lines"),
    "thermal_power_station": ("industrial blocks, cooling structures, and power facilities", "large industrial components with prominent circular and linear structures"),
    "wetland": ("shallow water, marsh vegetation, and channels", "interleaved irregular vegetation and narrow water patches"),
    # AID and UC-Merced categories not explicitly in NWPU-RESISC45:
    "bare_land": ("exposed soil, rock, and sparse ground cover", "continuous open land with little vegetation or built structure"),
    "center": ("central public buildings, plazas, and access roads", "a dense central building cluster around a focal open space"),
    "farmland": ("crop fields, paths, and agricultural parcels", "regular field subdivisions with long straight boundaries"),
    "park": ("grass, trees, footpaths, and recreational spaces", "connected green areas and curved paths within urban surroundings"),
    "playground": ("play areas, courts, paths, and equipment", "open recreational ground surrounded by built or vegetated features"),
    "pond": ("small enclosed water body, shoreline, and vegetation", "compact bounded water regions within land"),
    "resort": ("vacation buildings, pools, gardens, and roads", "clusters of low-rise buildings distributed among open amenities"),
    "school": ("educational buildings, courtyards, and athletic grounds", "several rectangular buildings organized around open yards"),
    "square": ("paved plaza, pedestrian paths, and surrounding buildings", "a large geometric open space framed by urban blocks"),
    "viaduct": ("elevated road or railway, pillars, and surrounding terrain", "a long straight or curved elevated transport corridor"),
    "agricultural": ("cultivated plots, fields, and farm access roads", "large repeating agricultural parcels separated by linear boundaries"),
    "buildings": ("building roofs, streets, and paved surroundings", "clusters of rectangular roofs along connected streets"),
}


def make_prompts(class_names: Sequence[str], *, override: str | Path | None = None,
                 strict: bool = True) -> list[tuple[str, str, str]]:
    """Build [identity, object-composition, geometry-layout] prompts per class."""
    attributes = dict(ATTRIBUTES)
    if override:
        data = json.loads(Path(override).read_text(encoding="utf-8"))
        for key, value in data.items():
            if not isinstance(value, dict) or not {"objects", "layout"} <= set(value):
                raise ValueError(f"Prompt override for {key} needs objects and layout")
            attributes[canonical_class_name(key)] = (value["objects"], value["layout"])

    result = []
    for original in class_names:
        key = canonical_class_name(original)
        name = key.replace("_", " ")
        article = "an" if name[0] in "aeiou" else "a"
        if key not in attributes:
            if strict:
                raise KeyError(
                    f"No structured prompts for {original!r}; provide --prompt-overrides "
                    "or intentionally enable --allow-generic-prompts."
                )
            objects = f"typical land cover and objects associated with {name}"
            layout = f"spatial structures characteristic of {name}"
        else:
            objects, layout = attributes[key]
        result.append((
            f"A remote sensing image of {article} {name}.",
            f"A {name} scene usually contains {objects}.",
            f"A {name} often shows {layout}.",
        ))
    return result


@torch.no_grad()
def encode_prompt_bank(backbone, class_names: Sequence[str], *,
                       override: str | Path | None = None,
                       strict: bool = True) -> Tensor:
    """Encode the static three-prompt bank once with frozen RemoteCLIP text."""
    prompt_sets = make_prompts(class_names, override=override, strict=strict)
    flat = [prompt for triplet in prompt_sets for prompt in triplet]
    embeddings = backbone.encode_texts(flat)
    return embeddings.reshape(len(class_names), 3, -1)
