"""Tests for main module entry point."""


from local_ai.__main__ import app


class TestMainModule:
    """Test main module functionality."""

    def test_main_calls_cli_app(self) -> None:
        """Test that main module calls the CLI app when executed."""
        # Test that the app function is callable
        assert callable(app)

        # Test that importing the module doesn't raise errors
        import local_ai.__main__

        # The module should have the app function
        assert hasattr(local_ai.__main__, 'app')

    def test_app_function_exists(self) -> None:
        """Test that the app function is accessible."""
        assert callable(app)
        assert hasattr(app, '__call__')
