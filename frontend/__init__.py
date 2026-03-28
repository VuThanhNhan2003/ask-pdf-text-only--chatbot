"""Frontend presentation layer for Streamlit UI."""
# frontend/__init__.py
from .styles import inject_global_css, render_app_header, render_sidebar_header
__all__ = ["inject_global_css", "render_app_header", "render_sidebar_header"]