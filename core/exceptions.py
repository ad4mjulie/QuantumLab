"""Domain-specific exceptions for the QuantumLab application."""

class QuantumLabError(Exception):
    """Base exception for all QuantumLab errors."""
    pass

class PhysicsError(QuantumLabError):
    """Errors related to physics simulations."""
    pass

class QuantumError(QuantumLabError):
    """Errors related to quantum algorithms."""
    pass

class ResourceNotFoundError(QuantumLabError):
    """Raised when an orbital or resource is not found."""
    pass

class ValidationError(QuantumLabError):
    """Raised when input parameters fail domain validation."""
    pass
