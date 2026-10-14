"""Billing helpers for the support plans."""


def compute_prorated_charge(monthly_price: float, days_used: int,
                            days_in_month: int = 30) -> float:
    """Charge for a partial month, rounded to the cent."""
    if days_in_month <= 0:
        raise ValueError("days_in_month must be positive")
    return round(monthly_price * days_used / days_in_month, 2)


def seats_over_allowance(seats_used: int, seats_included: int) -> int:
    """How many seats are billed as overage."""
    extra = seats_used - seats_included
    return extra if extra > 0 else 0


def annual_discount(monthly_price: float) -> float:
    """Two months free on an annual plan."""
    return round(monthly_price * 10, 2)
