import databento as db
import pandas as pd
import datetime

from finm37000 import (
    get_databento_api_key,
    temp_env,
    tz_chicago,
)

def init_client() -> db.Historical:
    """Create a databento Historical client with the API key loaded via temp_env"""
    with temp_env(DATABENTO_API_KEY=get_databento_api_key()):
        client = db.Historical()
    return client

def to_chicago(ts: datetime.date | pd.Timestamp) -> pd.Timestamp:
    """Convert date-like input to a tz-aware Chicago timestamp."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize(tz_chicago)
    return t.tz_convert(tz_chicago)