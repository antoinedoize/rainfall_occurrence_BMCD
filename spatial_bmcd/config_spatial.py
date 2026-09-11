"""Hyper-parameters specific to the spatial (multi-station) analysis.

The single-site pipeline is configured by ``article_code.util_files.config``
(paths, wet-day threshold, seasons); this module only holds the knobs that no
single-site script reads, so that ``article_code`` stays untouched by spatial
work. ``config`` is re-exported for convenience::

    from spatial_bmcd.config_spatial import config, MAX_NAN_FRACTION_STATION
"""

from article_code.util_files import config  # noqa: F401  (re-exported)


# ---------------------------------------------------------------------------
# Station admissibility
# ---------------------------------------------------------------------------

# A station whose RR column is missing more than this fraction of the days of
# the window below is dropped from the spatial station set; see
# :func:`spatial_bmcd.spatial_model.filter_stations_by_nan_fraction`.
#
# On the Spain--Portugal subset (35 stations with single-site fits), measured
# over 1980--2020: 0.05 keeps 8 stations, 0.10 keeps 16, 0.20 keeps 17,
# 0.50 keeps 26. The dropped tail is genuinely unusable -- LLEIDA-AJUNTAMENT
# has no observation at all inside the window.
MAX_NAN_FRACTION_STATION = 0.50

# Window the missing-day fraction is measured over. Deliberately the window of
# the spatial analysis itself (the YEAR_MIN / YEAR_MAX of the notebooks), not
# the full ECAD record: a station that only started recording in 1975 must not
# be penalised for the decades preceding it, and one that stopped in 2005 must
# be, since half of the analysis window is then empty.
NAN_FRACTION_YEAR_MIN = 1980
NAN_FRACTION_YEAR_MAX = 2020
