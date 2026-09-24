import pandas as pd


def test_valid_order_feature_ranges():
    df = pd.DataFrame(
        {
            "total_items": [1, 2, 3],
            "total_price": [10.0, 20.0, 30.0],
            "total_freight_value": [
                1.0,
                2.0,
                3.0,
            ],
            "unique_products": [
                1,
                2,
                3,
            ],
            "unique_sellers": [
                1,
                1,
                2,
            ],
        }
    )

    assert (
        df["total_items"] >= 0
    ).all()

    assert (
        df["total_price"] >= 0
    ).all()

    assert (
        df["total_freight_value"] >= 0
    ).all()

    assert (
        df["unique_products"] >= 0
    ).all()

    assert (
        df["unique_sellers"] >= 0
    ).all()