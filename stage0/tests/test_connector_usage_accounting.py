import pandas as pd


def test_connector_usage_accounting():
    routes = pd.DataFrame({
        "order_id": ["a", "a", "b"],
        "link_id": ["road", "connector", "road"],
    })
    used = routes.loc[routes.link_id.eq("connector")]
    assert used.order_id.nunique() == 1
    assert len(used) == 1
