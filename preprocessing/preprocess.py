import pandas as pd
import numpy as np
from pathlib import Path
from functools import lru_cache

DATASET_PATH = Path(__file__).parent.parent / "dataset" / "BirdLLM_dataset.csv"

# Common English name -> scientific name (or partial) for species in this dataset
COMMON_NAMES: dict[str, str] = {
    "american robin": "turdus migratorius",
    "robin": "turdus migratorius",
    "ruby-throated hummingbird": "archilochus colubris",
    "ruby throated hummingbird": "archilochus colubris",
    "hummingbird": "archilochus colubris",
    "hummingbirds": "archilochus colubris",
    "canada goose": "branta canadensis",
    "canadian goose": "branta canadensis",
    "tufted titmouse": "baeolophus bicolor",
    "red-shouldered hawk": "buteo lineatus",
    "red shouldered hawk": "buteo lineatus",
    "tree swallow": "tachycineta bicolor",
    "northern cardinal": "cardinalis cardinalis",
    "cardinal": "cardinalis cardinalis",
    "turkey vulture": "cathartes aura",
    "black vulture": "coragyps atratus",
    "blue jay": "cyanocitta cristata",
    "american crow": "corvus brachyrhynchos",
    "crow": "corvus brachyrhynchos",
    "common raven": "corvus corax",
    "fish crow": "corvus ossifragus",
    "red-winged blackbird": "agelaius phoeniceus",
    "red winged blackbird": "agelaius phoeniceus",
    "wood duck": "aix sponsa",
    "mallard": "anas platyrhynchos",
    "great blue heron": "ardea herodias",
    "great egret": "ardea alba",
    "great horned owl": "bubo virginianus",
    "bufflehead": "bucephala albeola",
    "red-tailed hawk": "buteo jamaicensis",
    "red tailed hawk": "buteo jamaicensis",
    "broad-winged hawk": "buteo platypterus",
    "broad winged hawk": "buteo platypterus",
    "swainson's hawk": "buteo swainsoni",
    "green heron": "butorides virescens",
    "cedar waxwing": "bombycilla cedrorum",
    "canada warbler": "cardellina canadensis",
    "wilson's warbler": "cardellina pusilla",
    "chimney swift": "chaetura pelagica",
    "killdeer": "charadrius vociferus",
    "semipalmated plover": "charadrius semipalmatus",
    "northern harrier": "circus cyaneus",
    "marsh wren": "cistothorus palustris",
    "sedge wren": "cistothorus platensis",
    "yellow-billed cuckoo": "coccyzus americanus",
    "black-billed cuckoo": "coccyzus erythropthalmus",
    "northern flicker": "colaptes auratus",
    "flicker": "colaptes auratus",
    "northern bobwhite": "colinus virginianus",
    "rock pigeon": "columba livia",
    "pigeon": "columba livia",
    "olive-sided flycatcher": "contopus cooperi",
    "eastern wood-pewee": "contopus virens",
    "american crow": "corvus brachyrhynchos",
    "sandhill crane": "grus canadensis",
    "sandhill cranes": "grus canadensis",
    "house finch": "haemorhous mexicanus",
    "purple finch": "haemorhous purpureus",
    "bald eagle": "haliaeetus leucocephalus",
    "eagle": "haliaeetus leucocephalus",
    "barn swallow": "hirundo rustica",
    "wood thrush": "hylocichla mustelina",
    "yellow-breasted chat": "icteria virens",
    "baltimore oriole": "icterus galbula",
    "oriole": "icterus galbula",
    "orchard oriole": "icterus spurius",
    "mississippi kite": "ictinia mississippiensis",
    "dark-eyed junco": "junco hyemalis",
    "junco": "junco hyemalis",
    "loggerhead shrike": "lanius ludovicianus",
    "ring-billed gull": "larus delawarensis",
    "herring gull": "larus smithsonianus",
    "hooded merganser": "lophodytes cucullatus",
    "belted kingfisher": "megaceryle alcyon",
    "kingfisher": "megaceryle alcyon",
    "eastern screech-owl": "megascops asio",
    "screech owl": "megascops asio",
    "red-bellied woodpecker": "melanerpes carolinus",
    "red bellied woodpecker": "melanerpes carolinus",
    "red-headed woodpecker": "melanerpes erythrocephalus",
    "wild turkey": "meleagris gallopavo",
    "turkey": "meleagris gallopavo",
    "swamp sparrow": "melospiza georgiana",
    "lincoln's sparrow": "melospiza lincolnii",
    "song sparrow": "melospiza melodia",
    "northern mockingbird": "mimus polyglottos",
    "mockingbird": "mimus polyglottos",
    "black-and-white warbler": "mniotilta varia",
    "brown-headed cowbird": "molothrus ater",
    "cowbird": "molothrus ater",
    "great crested flycatcher": "myiarchus crinitus",
    "yellow-crowned night-heron": "nyctanassa violacea",
    "black-crowned night-heron": "nycticorax nycticorax",
    "night heron": "nycticorax nycticorax",
    "osprey": "pandion haliaetus",
    "louisiana waterthrush": "parkesia motacilla",
    "northern waterthrush": "parkesia noveboracensis",
    "house sparrow": "passer domesticus",
    "indigo bunting": "passerina cyanea",
    "bunting": "passerina cyanea",
    "painted bunting": "passerina ciris",
    "blue grosbeak": "passerina caerulea",
    "double-crested cormorant": "phalacrocorax auritus",
    "cormorant": "phalacrocorax auritus",
    "rose-breasted grosbeak": "pheucticus ludovicianus",
    "grosbeak": "pheucticus ludovicianus",
    "eastern towhee": "pipilo erythrophthalmus",
    "towhee": "pipilo erythrophthalmus",
    "scarlet tanager": "piranga olivacea",
    "summer tanager": "piranga rubra",
    "tanager": "piranga",
    "pied-billed grebe": "podilymbus podiceps",
    "black-capped chickadee": "poecile atricapillus",
    "carolina chickadee": "poecile carolinensis",
    "chickadee": "poecile",
    "blue-gray gnatcatcher": "polioptila caerulea",
    "gnatcatcher": "polioptila caerulea",
    "purple martin": "progne subis",
    "martin": "progne subis",
    "prothonotary warbler": "protonotaria citrea",
    "common grackle": "quiscalus quiscula",
    "grackle": "quiscalus quiscula",
    "ruby-crowned kinglet": "regulus calendula",
    "golden-crowned kinglet": "regulus satrapa",
    "kinglet": "regulus",
    "eastern phoebe": "sayornis phoebe",
    "phoebe": "sayornis phoebe",
    "ovenbird": "seiurus aurocapilla",
    "rufous hummingbird": "selasphorus rufus",
    "northern parula": "setophaga americana",
    "black-throated blue warbler": "setophaga caerulescens",
    "bay-breasted warbler": "setophaga castanea",
    "cerulean warbler": "setophaga cerulea",
    "hooded warbler": "setophaga citrina",
    "yellow-rumped warbler": "setophaga coronata",
    "butter butt": "setophaga coronata",
    "prairie warbler": "setophaga discolor",
    "yellow-throated warbler": "setophaga dominica",
    "blackburnian warbler": "setophaga fusca",
    "magnolia warbler": "setophaga magnolia",
    "palm warbler": "setophaga palmarum",
    "chestnut-sided warbler": "setophaga pensylvanica",
    "yellow warbler": "setophaga petechia",
    "pine warbler": "setophaga pinus",
    "american redstart": "setophaga ruticilla",
    "blackpoll warbler": "setophaga striata",
    "cape may warbler": "setophaga tigrina",
    "black-throated green warbler": "setophaga virens",
    "warbler": "setophaga",
    "eastern bluebird": "sialia sialis",
    "bluebird": "sialia sialis",
    "red-breasted nuthatch": "sitta canadensis",
    "white-breasted nuthatch": "sitta carolinensis",
    "brown-headed nuthatch": "sitta pusilla",
    "nuthatch": "sitta",
    "northern shoveler": "spatula clypeata",
    "blue-winged teal": "spatula discors",
    "teal": "spatula discors",
    "yellow-bellied sapsucker": "sphyrapicus varius",
    "pine siskin": "spinus pinus",
    "american goldfinch": "spinus tristis",
    "goldfinch": "spinus tristis",
    "dickcissel": "spiza americana",
    "chipping sparrow": "spizella passerina",
    "field sparrow": "spizella pusilla",
    "barn swallow": "hirundo rustica",
    "cliff swallow": "petrochelidon pyrrhonota",
    "northern rough-winged swallow": "stelgidopteryx serripennis",
    "swallow": "hirundo",
    "eurasian collared-dove": "streptopelia decaocto",
    "dove": "zenaida macroura",
    "mourning dove": "zenaida macroura",
    "barred owl": "strix varia",
    "owl": "strix varia",
    "eastern meadowlark": "sturnella magna",
    "meadowlark": "sturnella magna",
    "european starling": "sturnus vulgaris",
    "starling": "sturnus vulgaris",
    "carolina wren": "thryothorus ludovicianus",
    "wren": "thryothorus ludovicianus",
    "brown thrasher": "toxostoma rufum",
    "thrasher": "toxostoma rufum",
    "lesser yellowlegs": "tringa flavipes",
    "greater yellowlegs": "tringa melanoleuca",
    "yellowlegs": "tringa",
    "house wren": "troglodytes aedon",
    "winter wren": "troglodytes hiemalis",
    "eastern kingbird": "tyrannus tyrannus",
    "kingbird": "tyrannus tyrannus",
    "barn owl": "tyto furcata",
    "golden-winged warbler": "vermivora chrysoptera",
    "blue-winged warbler": "vermivora cyanoptera",
    "yellow-throated vireo": "vireo flavifrons",
    "warbling vireo": "vireo gilvus",
    "white-eyed vireo": "vireo griseus",
    "red-eyed vireo": "vireo olivaceus",
    "vireo": "vireo",
    "white-throated sparrow": "zonotrichia albicollis",
    "white-crowned sparrow": "zonotrichia leucophrys",
    "sparrow": "melospiza melodia",
    "downy woodpecker": "dryobates pubescens",
    "hairy woodpecker": "leuconotopicus villosus",
    "pileated woodpecker": "dryocopus pileatus",
    "woodpecker": "melanerpes carolinus",
    "gray catbird": "dumetella carolinensis",
    "catbird": "dumetella carolinensis",
    "little blue heron": "egretta caerulea",
    "snowy egret": "egretta thula",
    "tricolored heron": "egretta tricolor",
    "egret": "ardea alba",
    "heron": "ardea herodias",
    "swallow-tailed kite": "elanoides forficatus",
    "american bittern": "botaurus lentiginosus",
    "bittern": "botaurus lentiginosus",
    "horned lark": "eremophila alpestris",
    "lark": "eremophila alpestris",
    "rusty blackbird": "euphagus carolinus",
    "blackbird": "agelaius phoeniceus",
    "merlin": "falco columbarius",
    "peregrine falcon": "falco peregrinus",
    "falcon": "falco peregrinus",
    "american kestrel": "falco sparverius",
    "kestrel": "falco sparverius",
    "american coot": "fulica americana",
    "coot": "fulica americana",
    "wilson's snipe": "gallinago delicata",
    "snipe": "gallinago delicata",
    "common gallinule": "gallinula chloropus",
    "common loon": "gavia immer",
    "loon": "gavia immer",
    "kentucky warbler": "geothlypis formosa",
    "mourning warbler": "geothlypis philadelphia",
    "common yellowthroat": "geothlypis trichas",
    "yellowthroat": "geothlypis trichas",
    "worm-eating warbler": "helmitheros vermivorum",
    "evening grosbeak": "hesperiphona vespertina",
    "anhinga": "anhinga anhinga",
    "bobolink": "dolichonyx oryzivorus",
    "upland sandpiper": "bartramia longicauda",
    "ruffed grouse": "bonasa umbellus",
    "grouse": "bonasa umbellus",
    "common redpoll": "acanthis flammea",
    "redpoll": "acanthis flammea",
    "sharp-shinned hawk": "accipiter striatus",
    "spotted sandpiper": "actitis macularius",
    "sandpiper": "actitis macularius",
    "northern saw-whet owl": "aegolius acadicus",
    "saw-whet owl": "aegolius acadicus",
    "grasshopper sparrow": "ammodramus savannarum",
    "northern pintail": "anas acuta",
    "pintail": "anas acuta",
    "green-winged teal": "anas crecca",
    "american black duck": "anas rubripes",
    "greater white-fronted goose": "anser albifrons",
    "snow goose": "anser caerulescens",
    "ross's goose": "anser rossii",
    "american pipit": "anthus rubescens",
    "chuck-will's-widow": "antrostomus carolinensis",
    "eastern whip-poor-will": "antrostomus vociferus",
    "whip-poor-will": "antrostomus vociferus",
    "limpkin": "aramus guarauna",
    "black-chinned hummingbird": "archilochus alexandri",
    "lesser scaup": "aythya affinis",
    "scaup": "aythya affinis",
    "redhead": "aythya americana",
    "ring-necked duck": "aythya collaris",
    "greater scaup": "aythya marila",
    "canvasback": "aythya valisineria",
    "upland sandpiper": "bartramia longicauda",
    "turnstone": "arenaria interpres",
    "ruddy turnstone": "arenaria interpres",
    "least sandpiper": "calidris minutilla",
    "dunlin": "calidris alpina",
    "sanderling": "calidris alba",
    "pectoral sandpiper": "calidris melanotos",
    "western sandpiper": "calidris mauri",
    "white-rumped sandpiper": "calidris fuscicollis",
    "baird's sandpiper": "calidris bairdii",
    "stilt sandpiper": "calidris himantopus",
    "semipalmated sandpiper": "calidris pusilla",
    "buff-breasted sandpiper": "calidris subruficollis",
    "orange-crowned warbler": "leiothlypis celata",
    "tennessee warbler": "leiothlypis peregrina",
    "nashville warbler": "leiothlypis ruficapilla",
    "laughing gull": "leucophaeus atricilla",
    "franklin's gull": "leucophaeus pipixcan",
    "gull": "larus delawarensis",
    "short-billed dowitcher": "limnodromus griseus",
    "long-billed dowitcher": "limnodromus scolopaceus",
    "dowitcher": "limnodromus",
    "swainson's warbler": "limnothlypis swainsonii",
    "red crossbill": "loxia curvirostra",
    "crossbill": "loxia curvirostra",
    "american wigeon": "mareca americana",
    "wigeon": "mareca americana",
    "gadwall": "mareca strepera",
    "common merganser": "mergus merganser",
    "red-breasted merganser": "mergus serrator",
    "merganser": "mergus merganser",
    "wood stork": "mycteria americana",
    "stork": "mycteria americana",
    "helmeted guineafowl": "numida meleagris",
    "common moorhen": "gallinula chloropus",
    "sora": "porzana carolina",
    "virginia rail": "rallus limicola",
    "rail": "rallus limicola",
    "american avocet": "recurvirostra americana",
    "avocet": "recurvirostra americana",
    "bank swallow": "riparia riparia",
    "black-legged kittiwake": "rissa tridactyla",
    "kittiwake": "rissa tridactyla",
    "american woodcock": "scolopax minor",
    "woodcock": "scolopax minor",
    "henslow's sparrow": "centronyx henslowii",
    "brown creeper": "certhia americana",
    "creeper": "certhia americana",
    "lark sparrow": "chondestes grammacus",
    "common nighthawk": "chordeiles minor",
    "nighthawk": "chordeiles minor",
    "black tern": "chlidonias niger",
    "tern": "sterna forsteri",
    "forster's tern": "sterna forsteri",
    "common tern": "sterna hirundo",
    "caspian tern": "hydroprogne caspia",
    "least tern": "sternula antillarum",
    "Bonaparte's gull": "chroicocephalus philadelphia",
    "long-tailed duck": "clangula hyemalis",
    "oldsquaw": "clangula hyemalis",
    "common goldeneye": "bucephala clangula",
    "goldeneye": "bucephala clangula",
    "surf scoter": "melanitta perspicillata",
    "black scoter": "melanitta americana",
    "white-winged scoter": "melanitta deglandi",
    "scoter": "melanitta americana",
    "tundra swan": "cygnus columbianus",
    "mute swan": "cygnus olor",
    "swan": "cygnus columbianus",
    "black-bellied whistling duck": "dendrocygna autumnalis",
    "whistling duck": "dendrocygna autumnalis",
    "red-necked phalarope": "phalaropus lobatus",
    "wilson's phalarope": "phalaropus tricolor",
    "phalarope": "phalaropus",
    "ring-necked pheasant": "phasianus colchicus",
    "pheasant": "phasianus colchicus",
    "purple gallinule": "porphyrio martinica",
    "gallinule": "porphyrio martinica",
    "white ibis": "eudocimus albus",
    "ibis": "plegadis falcinellus",
    "glossy ibis": "plegadis falcinellus",
    "white-faced ibis": "plegadis chihi",
    "american golden-plover": "pluvialis dominica",
    "plover": "charadrius semipalmatus",
    "horned grebe": "podiceps auritus",
    "red-necked grebe": "podiceps grisegena",
    "eared grebe": "podiceps nigricollis",
    "grebe": "podilymbus podiceps",
    "vesper sparrow": "pooecetes gramineus",
    "magnificent frigatebird": "fregata magnificens",
    "frigatebird": "fregata magnificens",
    "american white pelican": "pelecanus erythrorhynchos",
    "brown pelican": "pelecanus occidentalis",
    "pelican": "pelecanus erythrorhynchos",
    "northern gannet": "morus bassanus",
    "clay-colored sparrow": "spizella pallida",
    "american tree sparrow": "spizelloides arborea",
    "tree sparrow": "spizelloides arborea",
    "yellow-headed blackbird": "xanthocephalus xanthocephalus",
    "sabine's gull": "xema sabini",
    "white-winged dove": "zenaida asiatica",
    "peafowl": "pavo cristatus",
    "peacock": "pavo cristatus",
    "muscovy duck": "cairina moschata",
    "duck": "anas platyrhynchos",
    "egyptian goose": "alopochen aegyptiaca",
    "goose": "branta canadensis",
    "black-backed woodpecker": "picoides arcticus",
    "philadelphia vireo": "vireo philadelphicus",
    "blue-headed vireo": "vireo solitarius",
    "western kingbird": "tyrannus verticalis",
    "say's phoebe": "sayornis saya",
    "western wood-pewee": "empidonax difficilis",
    "yellow-bellied flycatcher": "empidonax flaviventris",
    "least flycatcher": "empidonax minimus",
    "willow flycatcher": "empidonax traillii",
    "alder flycatcher": "empidonax alnorum",
    "acadian flycatcher": "empidonax virescens",
    "flycatcher": "empidonax minimus",
    "fox sparrow": "passerella iliaca",
    "savannah sparrow": "passerculus sandwichensis",
}


def resolve_common_name(query: str) -> str:
    """Convert a common name query to a scientific name string if recognized."""
    import re
    # Strip parenthetical qualifiers e.g. "Hummingbird (general)" -> "hummingbird"
    cleaned = re.sub(r"\s*\(.*?\)", "", query).strip().lower()

    # Exact match
    if cleaned in COMMON_NAMES:
        return COMMON_NAMES[cleaned]

    # Partial match: check if any known common name is contained in the query
    for key, sci in COMMON_NAMES.items():
        if key in cleaned:
            return sci

    return cleaned

USECOLS = [
    "species", "genus", "locality",
    "decimalLatitude", "decimalLongitude",
    "eventDate", "individualCount",
]

DTYPES = {
    "species": "str",
    "genus": "str",
    "locality": "str",
    "decimalLatitude": "float32",
    "decimalLongitude": "float32",
    "individualCount": "float32",
}

_cached_df = None


def load_dataset(path=None) -> pd.DataFrame:
    """Load and cache the dataset with minimal memory footprint."""
    global _cached_df
    if _cached_df is not None and path is None:
        return _cached_df

    p = Path(path) if path else DATASET_PATH
    df = pd.read_csv(
        p,
        sep="\t",
        usecols=lambda c: c in USECOLS,
        dtype=DTYPES,
        low_memory=True,
    )

    df["eventDate"] = pd.to_datetime(df["eventDate"], errors="coerce", format="%Y-%m-%d")
    df = df.dropna(subset=["eventDate", "species"])
    df["species"] = df["species"].str.strip()

    iso = df["eventDate"].dt.isocalendar()
    df["week"] = iso.week.astype("int8")
    df["year"] = iso.year.astype("int16")
    # Collapse ISO week 53 into week 52
    df.loc[df["week"] == 53, "week"] = 52

    if path is None:
        _cached_df = df
    return df


def filter_by_species(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Case-insensitive partial match on species name, then genus.
    Resolves common English names to scientific names first."""
    q = resolve_common_name(query.strip())
    mask = df["species"].str.contains(q, case=False, na=False, regex=False)
    if not mask.any():
        mask = df["genus"].str.contains(q, case=False, na=False, regex=False)
    return df[mask]


def filter_by_location(
    df: pd.DataFrame,
    lat: float,
    lon: float,
    radius_deg: float = 1.0,
) -> pd.DataFrame:
    """Bounding box filter around a lat/lon center."""
    return df[
        df["decimalLatitude"].between(lat - radius_deg, lat + radius_deg)
        & df["decimalLongitude"].between(lon - radius_deg, lon + radius_deg)
    ]


def filter_by_locality_string(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Case-insensitive substring match on the locality field."""
    return df[df["locality"].str.contains(query.strip(), case=False, na=False, regex=False)]


def _full_week_frame() -> pd.DataFrame:
    return pd.DataFrame({"week": range(1, 53)})


def compute_weekly_frequencies(
    df_species: pd.DataFrame,
    df_location: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-week normalized sighting frequency.

    normalized_frequency = species_count / total_checklists_that_week

    Returns a 52-row DataFrame with columns:
        week, species_count, total_count, normalized_frequency
    """
    sp_counts = (
        df_species.groupby("week").size().rename("species_count").reset_index()
    )
    total_counts = (
        df_location.groupby("week").size().rename("total_count").reset_index()
    )

    freq = (
        _full_week_frame()
        .merge(total_counts, on="week", how="left")
        .merge(sp_counts, on="week", how="left")
        .fillna(0)
    )
    freq["normalized_frequency"] = np.where(
        freq["total_count"] > 0,
        freq["species_count"] / freq["total_count"],
        0.0,
    )
    return freq.astype({"week": int, "species_count": int, "total_count": int})


def compute_yearly_weekly_frequencies(
    df_species: pd.DataFrame,
    df_location: pd.DataFrame,
) -> dict:
    """
    Compute per-year, per-week normalized frequencies.

    Returns {year: [52 floats]}.
    """
    years = sorted(df_location["year"].unique())
    result = {}
    for year in years:
        sp_yr = df_species[df_species["year"] == year]
        loc_yr = df_location[df_location["year"] == year]
        if len(loc_yr) == 0:
            continue
        freq = compute_weekly_frequencies(sp_yr, loc_yr)
        result[int(year)] = [round(float(v), 6) for v in freq["normalized_frequency"]]
    return result


def get_peak_weeks(weekly_freq: pd.DataFrame, top_n: int = 3) -> list:
    """Return the top N peak weeks sorted by normalized_frequency descending."""
    peaks = (
        weekly_freq.sort_values("normalized_frequency", ascending=False)
        .head(top_n)[["week", "normalized_frequency", "species_count"]]
        .to_dict("records")
    )
    return peaks


def run_pipeline(
    species_query: str,
    location_query: str = None,
    lat: float = None,
    lon: float = None,
    radius_deg: float = 1.0,
    dataset_path=None,
    data = None
) -> dict:
    """
    Full preprocessing pipeline.

    Priority for location filtering:
        1. lat/lon bounding box (if provided)
        2. locality string match (if provided)
        3. No location filter

    Returns a dict with:
        species_found, total_records, weekly_frequencies,
        peak_weeks, yearly_data, location_note, error (if any)
    """
    df = data if data is not None else load_dataset(dataset_path)

    # --- Location filter ---
    location_note = None
    if lat is not None and lon is not None:
        df_location = filter_by_location(df, lat, lon, radius_deg)
        if len(df_location) == 0:
            df_location = df
            location_note = "No records near that location; showing all available data."
    elif location_query:
        df_location = filter_by_locality_string(df, location_query)
        if len(df_location) == 0:
            df_location = df
            location_note = f'Location "{location_query}" not found in dataset; showing all available data.'
    else:
        df_location = df

    # --- Species filter ---
    df_species = filter_by_species(df_location, species_query)

    if len(df_species) == 0:
        # Fallback: ignore location, search globally
        df_species = filter_by_species(df, species_query)
        resolved = resolve_common_name(species_query)
        if len(df_species) == 0:
            return {
                "error": f'No records found for "{species_query}" (tried "{resolved}"). '
                         f'Try a scientific name or a more specific common name.',
                "species_found": [],
                "total_records": 0,
            }
        df_location = df
        location_note = (
            f'Species not found at the specified location; '
            f'showing data from all available locations.'
        )

    weekly_freq = compute_weekly_frequencies(df_species, df_location)
    peak_weeks = get_peak_weeks(weekly_freq)
    yearly_data = compute_yearly_weekly_frequencies(df_species, df_location)

    return {
        "species_found": sorted(df_species["species"].unique().tolist()),
        "total_records": int(len(df_species)),
        "weekly_frequencies": [
            {
                "week": int(r["week"]),
                "species_count": int(r["species_count"]),
                "total_count": int(r["total_count"]),
                "normalized_frequency": round(float(r["normalized_frequency"]), 6),
            }
            for r in weekly_freq.to_dict("records")
        ],
        "peak_weeks": [
            {
                "week": int(p["week"]),
                "normalized_frequency": round(float(p["normalized_frequency"]), 6),
                "species_count": int(p["species_count"]),
            }
            for p in peak_weeks
        ],
        "yearly_data": yearly_data,
        "location_note": location_note,
    }
