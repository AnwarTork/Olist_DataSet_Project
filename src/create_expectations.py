from pathlib import Path

import great_expectations as gx

from src.gx_setup import get_gx_context


SUITE_NAME = "olist_ml_table_suite"


def create_suite():
    context = get_gx_context()

    try:
        suite = context.suites.get(
            SUITE_NAME
        )
    except Exception:
        suite = gx.ExpectationSuite(
            name=SUITE_NAME
        )

        suite = context.suites.add(
            suite
        )

    return context, suite


def add_expectations(
    context,
    suite,
):
    expectations = [
        gx.expectations.ExpectColumnToExist(
            column="order_id"
        ),

        gx.expectations.ExpectColumnToExist(
            column="customer_id"
        ),

        gx.expectations.ExpectColumnToExist(
            column="is_late"
        ),

        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="order_id"
        ),

        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="is_late"
        ),

        gx.expectations.ExpectColumnValuesToBeInSet(
            column="is_late",
            value_set=[0, 1],
        ),

        gx.expectations.ExpectColumnValuesToBeBetween(
            column="total_items",
            min_value=0
        ),

        gx.expectations.ExpectColumnValuesToBeBetween(
            column="total_price",
            min_value=0
        ),

        gx.expectations.ExpectColumnValuesToBeBetween(
            column="total_freight_value",
            min_value=0
        ),

        gx.expectations.ExpectColumnValuesToBeBetween(
            column="total_payment_value",
            min_value=0
        ),

        gx.expectations.ExpectColumnValuesToBeBetween(
            column="unique_products",
            min_value=0
        ),

        gx.expectations.ExpectColumnValuesToBeBetween(
            column="unique_sellers",
            min_value=0
        ),
    ]

    for expectation in expectations:
        suite.add_expectation(
            expectation
        )

    suite.save()

    return suite


if __name__ == "__main__":

    context, suite = create_suite()

    add_expectations(
        context,
        suite,
    )

    print(
        f"Created suite: {SUITE_NAME}"
    )