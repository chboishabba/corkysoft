import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.components.maps import _pydeck_frame


def test_pydeck_frame_deduplicates_duplicate_columns() -> None:
    df = pd.DataFrame([[1, 2, 3]], columns=["lat", "lon", "lat"])
    result = _pydeck_frame(df)

    assert list(result.columns) == ["lat", "lon"]
    assert result.iloc[0].to_dict() == {"lat": 1, "lon": 2}
