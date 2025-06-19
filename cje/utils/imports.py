"""Utilities for handling optional imports with clear error messages."""

import importlib
import sys
from typing import Any, Dict, List, Optional, Tuple


class ImportChecker:
    """Check and report on optional dependencies."""

    OPTIONAL_DEPS = {
        "xgboost": {
            "package": "xgboost",
            "install": "pip install xgboost",
            "features": ["XGBoost outcome models for causal estimation"],
        },
        "labelbox": {
            "package": "labelbox",
            "install": "pip install labelbox",
            "features": ["Labelbox integration for human labeling"],
        },
        "python-dotenv": {
            "package": "dotenv",
            "import_name": "dotenv",
            "install": "pip install python-dotenv",
            "features": ["Automatic .env file loading"],
        },
        "omegaconf": {
            "package": "omegaconf",
            "install": "pip install omegaconf",
            "features": ["Advanced configuration serialization"],
        },
        "together": {
            "package": "together",
            "install": "pip install together",
            "features": ["Together AI provider support"],
        },
    }

    @classmethod
    def check_all(cls) -> Dict[str, bool]:
        """Check all optional dependencies and return their status."""
        results = {}
        for name, info in cls.OPTIONAL_DEPS.items():
            import_name = str(info.get("import_name", info["package"]))
            try:
                importlib.import_module(import_name)
                results[name] = True
            except ImportError:
                results[name] = False
        return results

    @classmethod
    def require(cls, package: str, feature: str) -> Any:
        """
        Import a required package with a clear error message if missing.

        Args:
            package: The package to import
            feature: Description of what feature needs this package

        Returns:
            The imported module

        Raises:
            ImportError: With a helpful message about how to install
        """
        info = cls.OPTIONAL_DEPS.get(package, {})
        import_name = str(info.get("import_name", package))
        install_cmd = info.get("install", f"pip install {package}")

        try:
            return importlib.import_module(import_name)
        except ImportError as e:
            raise ImportError(
                f"\n{feature} requires '{package}' which is not installed.\n"
                f"Install it with:\n    {install_cmd}\n"
                f"Original error: {e}"
            ) from e

    @classmethod
    def optional_import(
        cls, package: str, feature: Optional[str] = None, warn: bool = True
    ) -> Tuple[Optional[Any], bool]:
        """
        Try to import an optional package.

        Args:
            package: The package to import
            feature: Optional description of what feature needs this
            warn: Whether to print a warning if import fails

        Returns:
            Tuple of (module or None, success bool)
        """
        info = cls.OPTIONAL_DEPS.get(package, {})
        import_name = str(info.get("import_name", package))
        install_cmd = info.get("install", f"pip install {package}")

        try:
            module = importlib.import_module(import_name)
            return module, True
        except ImportError:
            if warn:
                feature_desc = feature or f"features requiring {package}"
                print(
                    f"Warning: {feature_desc} will not be available.\n"
                    f"To enable, install with: {install_cmd}",
                    file=sys.stderr,
                )
            return None, False

    @classmethod
    def print_status(cls) -> None:
        """Print a status report of all optional dependencies."""
        print("\nCJE Optional Dependencies Status:")
        print("-" * 50)

        statuses = cls.check_all()
        for name, info in cls.OPTIONAL_DEPS.items():
            status = "✓ Installed" if statuses[name] else "✗ Not installed"
            print(f"\n{name}: {status}")
            print(f"  Features: {', '.join(info['features'])}")
            if not statuses[name]:
                print(f"  Install: {info['install']}")


def require_import(package: str, feature: str) -> Any:
    """Convenience function for required imports."""
    return ImportChecker.require(package, feature)


def optional_import(
    package: str, feature: Optional[str] = None, warn: bool = True
) -> Tuple[Optional[Any], bool]:
    """Convenience function for optional imports."""
    return ImportChecker.optional_import(package, feature, warn)
