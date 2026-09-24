"""Print the splits, columns and a sample of candidate hard-replay datasets (exploration only)."""

from datasets import get_dataset_config_names, load_dataset

CANDIDATES = [
    ("nikhilweee/sharc_modified", None),
    ("presencesw/contract-nli", None),
    ("reuben256/contract-nli", None),
    ("deepmind/aqua_rat", "raw"),
    ("allenai/math_qa", None),
]
OLD = [
    ("UCLNLP/sharc", None),
    ("kiddothe2b/contract-nli", None),
    ("emozilla/quality", None),
    ("facebook/anli", None),
    ("ibm-research/tab_fact", None),
    ("table-benchmark/tabfact", None),
    ("lukaemon/bbh", "date_understanding"),
]

for name, config in CANDIDATES:
    try:
        configs = get_dataset_config_names(name)
        ds = load_dataset(name, config or (configs[0] if configs else None))
        split = next(iter(ds))
        row = ds[split][0]
        print(f"== {name} [{config or (configs[0] if configs else '')}] configs={configs[:6]}")
        print("   splits:", {k: len(v) for k, v in ds.items()})
        print("   columns:", ds[split].column_names)
        print("   sample:", {k: (str(v)[:160]) for k, v in row.items()})
    except Exception as e:  # noqa: BLE001
        print(f"== {name}: FAILED {type(e).__name__}: {str(e)[:200]}")
