"""
model/src/preprocessor.py
──────────────────────────
Handles CSV loading, cleaning, and feature engineering.
Produces a feature matrix ready for cosine similarity computation.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """
    Loads and transforms raw item data into a normalised feature matrix.

    Feature engineering strategy
    ─────────────────────────────
    • Text features  (genres, director, cast) → TF-IDF vectors
    • Numeric features (rating, votes, year)   → MinMax-scaled
    • Final matrix = concat(tfidf_matrix, scaled_numeric)
    """

    # Columns expected in the raw CSV
    TEXT_COLS = ["genres", "director", "cast"]
    NUMERIC_COLS = ["rating", "votes", "year"]
    TITLE_COL = "title"
    ID_COL = "movie_id"

    def __init__(
        self,
        data_path: str | Path,
        tfidf_max_features: int = 5000,
        text_weight: float = 0.7,
        numeric_weight: float = 0.3,
    ) -> None:
        self.data_path = Path(data_path)
        self.tfidf_max_features = tfidf_max_features
        self.text_weight = text_weight
        self.numeric_weight = numeric_weight

        self.df: pd.DataFrame | None = None
        self.feature_matrix: np.ndarray | None = None
        self._tfidf = TfidfVectorizer(max_features=tfidf_max_features, ngram_range=(1, 2))
        self._scaler = MinMaxScaler()

    # ── Public interface ────────────────────────────────────────

    def load(self) -> DataPreprocessor:
        """Load CSV from disk and validate required columns."""
        logger.info("Loading dataset", path=str(self.data_path))
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        self.df = pd.read_csv(self.data_path)
        self._validate()
        logger.info("Dataset loaded", rows=len(self.df))
        return self

    def preprocess(self) -> DataPreprocessor:
        """Run full pipeline: clean → engineer → build matrix."""
        if self.df is None:
            raise RuntimeError("Call .load() before .preprocess()")

        self._clean()
        self._engineer_text_features()
        self._build_feature_matrix()
        logger.info("Preprocessing complete", shape=self.feature_matrix.shape)
        return self

    @property
    def titles(self) -> list[str]:
        return self.df[self.TITLE_COL].tolist()

    @property
    def records(self) -> list[dict]:
        return self.df.to_dict(orient="records")

    def get_item_index(self, title: str) -> int:
        """Return row index for a given title (case-insensitive)."""
        lower = self.df[self.TITLE_COL].str.lower()
        mask = lower == title.lower()
        if not mask.any():
            raise ValueError(f"Item not found: '{title}'")
        return int(mask.idxmax())

    def search_titles(self, query: str, limit: int = 10) -> list[str]:
        """Fuzzy search titles containing the query substring."""
        q = query.lower()
        matched = self.df[self.df[self.TITLE_COL].str.lower().str.contains(q, na=False)]
        return matched[self.TITLE_COL].head(limit).tolist()

    def get_categories(self) -> list[str]:
        """Return unique genre values across all items."""
        all_genres: set[str] = set()
        for genres_str in self.df["genres"].dropna():
            for g in genres_str.split("|"):
                all_genres.add(g.strip())
        return sorted(all_genres)

    # ── Private helpers ─────────────────────────────────────────

    def _validate(self) -> None:
        required = {self.TITLE_COL} | set(self.TEXT_COLS) | set(self.NUMERIC_COLS)
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns in dataset: {missing}")

    def _clean(self) -> None:
        logger.debug("Cleaning data")
        # Fill missing text fields
        for col in self.TEXT_COLS:
            self.df[col] = self.df[col].fillna("").astype(str)

        # Fill missing numerics with column median
        for col in self.NUMERIC_COLS:
            median = self.df[col].median()
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(median)

        # Remove duplicate titles (keep first)
        before = len(self.df)
        self.df.drop_duplicates(subset=self.TITLE_COL, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        logger.debug("Deduplication", removed=before - len(self.df))

    def _engineer_text_features(self) -> None:
        """Build a single 'soup' column from all text features."""
        def normalise(text: str) -> str:
            text = text.lower()
            text = re.sub(r"[^a-z0-9\s]", " ", text)
            return " ".join(text.split())

        # Pipe-separated values → space-separated tokens
        def expand_pipe(text: str) -> str:
            return " ".join(part.strip().replace(" ", "_") for part in text.split("|"))

        self.df["_soup"] = (
            self.df["genres"].apply(expand_pipe).apply(normalise) + " "
            + self.df["director"].apply(normalise) + " "
            + self.df["cast"].apply(expand_pipe).apply(normalise)
        )

    def _build_feature_matrix(self) -> None:
        """Combine TF-IDF text matrix with scaled numeric features."""
        # Text component
        tfidf_matrix = self._tfidf.fit_transform(self.df["_soup"]).toarray()
        tfidf_matrix = tfidf_matrix * self.text_weight

        # Numeric component
        numeric_matrix = self._scaler.fit_transform(self.df[self.NUMERIC_COLS].values)
        numeric_matrix = numeric_matrix * self.numeric_weight

        self.feature_matrix = np.hstack([tfidf_matrix, numeric_matrix])
