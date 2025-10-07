# auth/__init__.py
"""Authentication package"""
from auth.authentication import AuthService, AuthenticationError

__all__ = ['AuthService', 'AuthenticationError']