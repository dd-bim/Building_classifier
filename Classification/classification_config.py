"""
Gemeinsame Konfiguration für die Klassifikationspipeline.

Dieses Modul ist die einzige Quelle für:
- Feature-Gruppen (CORE_FEATURES, etc.)
- Klassifikationshierarchie (LEVELS)
- Mengen spezieller Levels (ENDLEVELS_DIRECT_AGE, SKIP_METRIC_LEVELS)

Importiert von: model_trainer, validate_data, classify_data
"""
from typing import List, Set, Tuple

# ---------------------------------------------------------------------------
# Feature-Gruppen
# ---------------------------------------------------------------------------
CORE_FEATURES: List[str] = [
    'roof_type', 'storeys_above_ground', 'building_footprint',
    'roof_ridge_height', 'eaves_height', 'storey_height',
    'number_roof_surfaces', 'roof_slope', 'development_type_code',
    'building_age',
]

SIMPLE_GEOM_FEATURES: List[str] = [
    'length_footprint',
    'width_footprint',
    'building_volume',
]

ADV_GEOM_FEATURES: List[str] = [
    'compactness',
    'convexity',
    'rectangularity',
]

NEIGH_FEATURES: List[str] = [
    'neighbour_density',
    'neighbour_avg_size',
    'neighbour_min_distance',
    'neighbour_majority_class',
]

RATIO_FEATURES: List[str] = [
    'ground_area_per_storey',
    'height_to_area_ratio',
    'footprint_ratio',
    'roof_height_ratio',
]

ALL_FEATURES: List[str] = (
    CORE_FEATURES + SIMPLE_GEOM_FEATURES + ADV_GEOM_FEATURES + NEIGH_FEATURES + RATIO_FEATURES
)

CATEGORICAL: List[str] = [
    'roof_type',
    'development_type_code',
    'neighbour_majority_class',
    'building_age',
]

# ---------------------------------------------------------------------------
# Klassifikationshierarchie
# ---------------------------------------------------------------------------
LEVELS: List[Tuple[str, List[str]]] = [
    ('1',    ['M', 'E', 'Other']),
    ('11',   ['MR', 'ME', 'ER', 'EE']),                                              # regelbasiert
    ('12',   ['HH', 'LW']),
    ('121',  ['HH3', 'HH4']),                                                        # Endlevel
    ('122',  ['LW1', 'LW2', 'LW3', 'LW7']),                                         # Endlevel
    ('112',  ['ME2', 'ME3', 'ME4', 'ME5', 'ME6', 'ME7']),                           # Endlevel
    ('113',  ['ER2', 'ER3', 'ER4', 'ER5', 'ER7']),                                  # Endlevel
    ('114',  ['EE1', 'EE2', 'EE3', 'EE4', 'EE5', 'EE7']),                          # Endlevel
    ('111',  ['MR2', 'MR3', 'MR4', 'MR5', 'MR6', 'MR7']),                          # Endlevel
    ('1111', ['MRO2', 'MRO3', 'MRO4', 'MRO7', 'MRG2', 'MRG3', 'MRG4', 'MRG7']),  # regelbasiert
]

# Endlevels mit direkter Baualterszuweisung (kein Modell bei vorhandenem building_age)
ENDLEVELS_DIRECT_AGE: Set[str] = {'111', '112', '113', '114', '121', '122'}

# Levels ohne Modelle (keine Metriken berechnet oder exportiert)
SKIP_METRIC_LEVELS: Set[str] = {'11', '1111'}
