
from pathlib import Path


###Paths config file.
# Only ECAD_RAW_DIR needs updating if using new data.

ROOT = Path(__file__).resolve().parent.parent.parent
# REPO_ROOT = ROOT.parent.parent  # rainfall_repo/
# LEGACY_REPO = REPO_ROOT / "one_sided_reflected_geometric_brownian_motion"

ECAD_RAW_DIR = ROOT / "data" / "ecad_data" / "ecad_data_south_europe_filtered"
STATION_METADATA_CSV = ROOT / "data" / "ecad_data" / "all_stations_metadata_filtered.csv"

EXPORTS_JSON_DIR = ROOT / "data" / "ecad_data" / "exports_json"
RESULTS_FIT_DIR  = ROOT / "results_fit"

### Station to treat
STATION_EXAMPLE = "PALERMO"
FIGURES_DIR = ROOT / "figures" /  STATION_EXAMPLE # companion_code_article/figures/
FIGURES_DIR.mkdir(exist_ok=True)


def set_active_station(name: str) -> Path:
    """Switch the active example station and (re)point FIGURES_DIR at it.

    Used by the per-station batch driver so that any plotting helper that reads
    `config.STATION_EXAMPLE` / `config.FIGURES_DIR` at call time (rather than
    at import time) picks up the new station without the module having to be
    re-imported.
    """
    global STATION_EXAMPLE, FIGURES_DIR
    STATION_EXAMPLE = name
    FIGURES_DIR = ROOT / "figures" / name
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR

# Other config
### Data processing
WET_DAY_THRESHOLD = 6
SEASONS = ("spring", "summer", "autumn", "winter")
LIST_MONTH_SEASONS = [
    ["01", "02", "03", "all","04", "05", "06", "07", "08", "09", "10", "11", "12"],
    ["03", "04", "05", "spring"],
    ["06", "07", "08", "summer"],
    ["09", "10", "11", "autumn"],
    ["12", "01", "02", "winter"],
]
START_YEAR = 1946
DROP_FIRST_LAST = True
MIN_NUMBER_SPELL = 365*10


