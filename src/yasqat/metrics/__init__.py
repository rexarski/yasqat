"""Distance metrics for sequence comparison."""

from yasqat.metrics.base import DistanceMatrix, build_substitution_matrix
from yasqat.metrics.chi2 import chi2_distance
from yasqat.metrics.dhd import dhd_distance
from yasqat.metrics.dtw import dtw_distance
from yasqat.metrics.engine import (
    METRICS,
    MetricSpec,
    compute_distance_matrix,
    pool_substitution_matrix,
)
from yasqat.metrics.euclidean import euclidean_distance
from yasqat.metrics.hamming import hamming_distance
from yasqat.metrics.lcp import lcp_distance, lcp_length, lcp_similarity
from yasqat.metrics.lcs import lcs_distance, lcs_length, lcs_similarity
from yasqat.metrics.nms import nms_distance, nmsmst_distance, svrspell_distance
from yasqat.metrics.om_variants import (
    om_boundary_weighted_distance,
    om_spell_scaled_distance,
    om_transition_penalty_distance,
)
from yasqat.metrics.omloc import omloc_distance
from yasqat.metrics.omspell import omspell_distance
from yasqat.metrics.omstran import omstran_costs, omstran_distance
from yasqat.metrics.optimal_matching import optimal_matching_distance
from yasqat.metrics.quality import (
    cluster_quality,
    distance_to_center,
    silhouette_score,
    silhouette_scores,
)
from yasqat.metrics.rlcp import rlcp_distance, rlcp_length, rlcp_similarity
from yasqat.metrics.softdtw import softdtw_distance
from yasqat.metrics.twed import twed_distance

__all__ = [
    "METRICS",
    "DistanceMatrix",
    "MetricSpec",
    "build_substitution_matrix",
    "chi2_distance",
    "cluster_quality",
    "compute_distance_matrix",
    "dhd_distance",
    "distance_to_center",
    "dtw_distance",
    "euclidean_distance",
    "hamming_distance",
    "lcp_distance",
    "lcp_length",
    "lcp_similarity",
    "lcs_distance",
    "lcs_length",
    "lcs_similarity",
    "nms_distance",
    "nmsmst_distance",
    "om_boundary_weighted_distance",
    "om_spell_scaled_distance",
    "om_transition_penalty_distance",
    "omloc_distance",
    "omspell_distance",
    "omstran_costs",
    "omstran_distance",
    "optimal_matching_distance",
    "pool_substitution_matrix",
    "rlcp_distance",
    "rlcp_length",
    "rlcp_similarity",
    "silhouette_score",
    "silhouette_scores",
    "softdtw_distance",
    "svrspell_distance",
    "twed_distance",
]
