"""CSV dataset loader for custom data files."""

import pandas as pd
from pathlib import Path
from typing import Iterable, Optional, Dict, Any

from .base import CJEDataset
from .schema import CJESample
from cje.utils.progress import track


class CSVDataset(CJEDataset):
    """Dataset loader for CSV files.

    This allows users to load their own data files in CSV format.
    Supports both CSV files and pandas DataFrames directly.

    Required columns: context
    Optional columns: uid, response, y_true, logp, plus any others (stored in meta)

    Example usage:
        # Load from CSV file
        dataset = CSVDataset("/path/to/data.csv")

        # Load from DataFrame
        dataset = CSVDataset.from_dataframe(df)

        # Iterate over samples
        for sample in dataset.itersamples():
            print(sample.context, sample.response)
    """

    file_path: Optional[Path]
    name: str
    pandas_kwargs: Dict[str, Any]
    _df: pd.DataFrame

    def __init__(self, file_path: str, **pandas_kwargs: Any):
        """Initialize with path to CSV file.

        Args:
            file_path: Path to the CSV file
            **pandas_kwargs: Additional arguments passed to pd.read_csv()
        """
        self.file_path = Path(file_path)
        self.name = str(file_path)
        self.pandas_kwargs = pandas_kwargs

        # Check file extension first
        if not self.file_path.suffix.lower() in [".csv", ".tsv"]:
            raise ValueError(f"File must have .csv or .tsv extension: {file_path}")

        # Then check if file exists
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        # Load and validate the DataFrame
        self._df = self._load_dataframe()

    def __init_from_dataframe__(
        self, df: pd.DataFrame, name: str = "dataframe"
    ) -> None:
        """Initialize directly from a pandas DataFrame."""
        self.file_path: Optional[Path] = None
        self.name = name
        self.pandas_kwargs = {}
        self._df = df.copy()
        self._validate_dataframe()

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, name: str = "dataframe") -> "CSVDataset":
        """Create dataset from pandas DataFrame.

        Args:
            df: Pandas DataFrame
            name: Name for the dataset

        Returns:
            CSVDataset instance
        """
        instance = cls.__new__(cls)
        instance.__init_from_dataframe__(df, name)
        return instance

    @classmethod
    def download(
        cls, cache_dir: Optional[str] = None, split: str = "train"
    ) -> "CSVDataset":
        """Not applicable for CSV files - use constructor directly."""
        raise NotImplementedError(
            "CSVDataset loads from local files. Use CSVDataset(file_path) directly."
        )

    def _load_dataframe(self) -> pd.DataFrame:
        """Load CSV file into DataFrame."""
        if self.file_path is None:
            raise RuntimeError("Cannot load DataFrame: no file path specified")

        try:
            # Auto-detect separator for .tsv files
            if self.file_path.suffix.lower() == ".tsv":
                sep = self.pandas_kwargs.get("sep", "\t")
                self.pandas_kwargs = {**self.pandas_kwargs, "sep": sep}

            df = pd.read_csv(self.file_path, **self.pandas_kwargs)
            self._validate_dataframe(df)
            return df
        except Exception as e:
            raise RuntimeError(f"Error loading CSV file {self.file_path}: {e}")

    def _validate_dataframe(self, df: Optional[pd.DataFrame] = None) -> None:
        """Validate that DataFrame has required columns."""
        if df is None:
            df = self._df

        # Check required columns
        if "context" not in df.columns:
            raise ValueError(
                f"CSV must contain 'context' column. Found columns: {list(df.columns)}"
            )

        # Validate data types
        if not df["context"].dtype == "object":
            raise ValueError("'context' column must contain string data")

    def itersamples(self) -> Iterable[CJESample]:
        """Lazily yield CJESample objects from the DataFrame.

        Yields:
            CJESample objects representing individual examples.
        """
        # Create a list of rows for progress tracking
        rows = list(self._df.iterrows())

        for idx, row in track(
            rows, description=f"Loading {self.name}", total=len(self._df)
        ):
            # Auto-generate uid if missing
            uid = str(row.get("uid", f"sample_{idx}"))

            # Required field
            context = str(row["context"])

            # Optional fields
            response = str(row.get("response", ""))
            y_true = row.get("y_true")
            logp = row.get("logp")

            # Convert NaN to None for optional fields
            if pd.isna(y_true):
                y_true = None
            if pd.isna(logp):
                logp = None

            # Store extra columns in meta
            meta: Dict[str, Any] = {}
            standard_cols = {"uid", "context", "response", "y_true", "logp"}
            for col in self._df.columns:
                if col not in standard_cols and not pd.isna(row[col]):
                    meta[col] = row[col]

            yield CJESample(
                uid=uid,
                context=context,
                response=response,
                y_true=y_true,
                logp=logp,
                meta=meta,
            )

    def __len__(self) -> int:
        """Get number of samples in the dataset."""
        return len(self._df)

    def __str__(self) -> str:
        name = self.file_path if self.file_path else self.name
        return f"CSVDataset({name})"

    def __repr__(self) -> str:
        if self.file_path:
            return f"CSVDataset(file_path='{self.file_path}')"
        else:
            return f"CSVDataset(dataframe, name='{self.name}')"
