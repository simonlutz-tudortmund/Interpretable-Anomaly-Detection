from decimal import ROUND_FLOOR, Decimal, getcontext


def get_bounds(x, bound_precision: int):
    # Convert inputs to Decimal early for precision
    x = Decimal(str(x))

    assert Decimal("0") < x < Decimal("1"), "x must be between 0 and 1 (exclusive)"

    # Set precision higher than needed to avoid intermediate rounding errors
    getcontext().prec = 50

    # Round x to that number of places
    rounded = x.quantize(
        Decimal("1e-{0}".format(bound_precision)), rounding=ROUND_FLOOR
    )

    if rounded > x:
        lower = rounded - Decimal("1e-{0}".format(bound_precision))
        upper = rounded
    else:
        lower = rounded
        upper = rounded + Decimal("1e-{0}".format(bound_precision))

    assert lower <= x <= upper, "Bounds are not valid"
    return lower, upper
