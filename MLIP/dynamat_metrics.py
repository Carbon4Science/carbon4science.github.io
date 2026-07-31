"""Static Matbench Discovery metrics for the dynamat benchmark.

The values are intentionally kept separate from the MD runner.  They can be
filled in from the project's agreed Matbench Discovery table without changing
the simulation or result schema.
"""

# Canonical names used by the new dynamat runner.
MODEL_NAMES = (
    "eSEN",
    "ORB",
    "DPA4",
    "NequIP",
    "MACE",
    "SevenNet",
    "Nequix",
    "CHGNet",
)

# Enter the CMDS values here when ready, for example:
#   "eSEN": {"CMDS": 0.123456},
# Keeping the entries explicit prevents accidental fallback to old CPS or
# single-structure RDF/MSD values.
MATBENCH_DISCOVERY_METRICS = {
    model: {"CMDS": None} for model in MODEL_NAMES
}


def get_metrics(model_name):
    """Return the hard-coded Matbench metrics for a canonical model name."""
    if model_name not in MATBENCH_DISCOVERY_METRICS:
        raise KeyError(f"No dynamat metric entry for {model_name}")
    return dict(MATBENCH_DISCOVERY_METRICS[model_name])
