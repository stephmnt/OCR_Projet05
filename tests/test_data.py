import pandas as pd

from projet_05 import dataset as ds


def test_clean_text_values_normalizes_tokens():
    df = pd.DataFrame({"col": ["nan", "JE ne sais pas", " value "]})
    cleaned = ds.clean_text_values(df)
    assert cleaned["col"].tolist() == [pd.NA, pd.NA, "value"]


def test_harmonize_id_column_extracts_digits_and_handles_invalid():
    df = pd.DataFrame({"id": ["EMP-001", "missing"]})
    harmonized = ds._harmonize_id_column(df, "id")
    assert harmonized["id"].tolist() == [1, pd.NA]
