import pandas as pd


def query_as_dataframe(cur, sql: str, params=None) -> pd.DataFrame:
    """Executes a SQL query and returns the result as a pandas DataFrame."""
    if params is not None:
        cur.execute(sql, params)
    else:
        cur.execute(sql)
    rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=colnames)
