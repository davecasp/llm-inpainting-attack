from datasets import Dataset, load_dataset


def load_data(dataset_name: str) -> Dataset:
    """Load and normalize a dataset into a Hugging Face Dataset with `goal` and `target` columns.

    **Arguments:**

    - `dataset_name`: Either a known dataset name (`"JailbreakBench"`, `"AdvBench"`) or a
        path to a CSV file containing `goal` and `target` columns.

    **Returns:**

    A Hugging Face `Dataset` containing at least `goal` and `target` columns.

    **Raises:**

    A `ValueError` if the dataset name is not recognized or required columns are missing.
    """
    if dataset_name == "JailbreakBench":
        dataset_dict = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
        dataset: Dataset = dataset_dict["harmful"]  # type: ignore
        dataset = dataset.rename_column("Goal", "goal")
        dataset = dataset.rename_column("Target", "target")
    elif dataset_name == "AdvBench":
        dataset_dict = load_dataset("walledai/AdvBench")
        dataset = dataset_dict["train"]  # type: ignore
        dataset = dataset.rename_column("prompt", "goal")
        if "target" not in dataset.column_names:
            raise ValueError("AdvBench dataset must contain a 'target' column")
    elif dataset_name.endswith(".csv"):
        dataset_dict = load_dataset("csv", data_files=dataset_name)
        dataset = dataset_dict["train"]  # type: ignore
        missing = [c for c in ("goal", "target") if c not in dataset.column_names]
        if len(missing) > 0:
            raise ValueError(f"CSV file is missing required columns: {missing}")
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Expected 'JailbreakBench', 'AdvBench', or a .csv path.")

    return dataset
