"""NLP.py-specific exceptions."""


class UserExitRequest(Exception):
    """Exception that the caller can use to request clean exit."""

    pass


class EqualityConstraintsError(Exception):
    """Exception that signals a problem with equality constraints."""

    pass


class InequalityConstraintsError(Exception):
    """Exception that signals a problem with inequality constraints."""

    pass


class BoundConstraintsError(Exception):
    """Exception that signals a problem with bound constraints."""

    pass


class GeneralConstraintsError(Exception):
    """Exception that signals a problem with general constraints."""

    pass


class InfeasibleError(Exception):
    """Error that can be raised to signal an infeasible iterate."""

    pass


class ShapeError(Exception):
    """Error that can be raised to signal a dimension mismatch."""

    pass


class LineSearchFailure(Exception):
    """Exception raised when a linesearch fails."""

    pass
