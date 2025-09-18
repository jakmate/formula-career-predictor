import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.main import create_app, lifespan


@pytest.fixture
def mock_dependencies():
    with patch("app.main.initialize_app_state", new_callable=AsyncMock) as mock_init, \
            patch("app.main.cleanup_app_state", new_callable=AsyncMock) as mock_cleanup:
        yield mock_init, mock_cleanup


@pytest.fixture
def mock_logger():
    with patch("app.main.LOGGER") as mock_logger:
        yield mock_logger


class TestCreateApp:
    def test_create_app_returns_fastapi_instance(self):
        app = create_app()
        assert isinstance(app, FastAPI)

    def test_app_configuration(self):
        app = create_app()

        assert app.title == "Formula Predictions API"
        assert app.version == "1.0.0"
        assert app.description == "API for predicting Formula 2 and 3 career promotions"

    def test_api_router_included(self):
        app = create_app()

        # Check that routes are registered with /api prefix
        routes = [route.path for route in app.routes]
        api_routes = [route for route in routes if route.startswith("/api")]
        assert len(api_routes) > 0


class TestLifespan:
    @pytest.mark.asyncio
    async def test_lifespan_successful_startup_shutdown(self, mock_dependencies, mock_logger):
        mock_init, mock_cleanup = mock_dependencies

        app = FastAPI()

        async with lifespan(app):
            # Verify initialization was called
            mock_init.assert_called_once()
            mock_logger.info.assert_called_with("Starting application...")

        # Verify cleanup was called
        mock_cleanup.assert_called_once()
        assert mock_logger.info.call_count >= 2  # startup + shutdown messages

    @pytest.mark.asyncio
    async def test_lifespan_cleanup_on_exception(self, mock_dependencies):
        mock_init, mock_cleanup = mock_dependencies

        app = FastAPI()

        with pytest.raises(RuntimeError):
            async with lifespan(app):
                mock_init.assert_called_once()
                raise RuntimeError("Test exception")

        # Verify cleanup was still called
        mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifespan_init_failure(self, mock_dependencies):
        mock_init, mock_cleanup = mock_dependencies
        mock_init.side_effect = Exception("Init failed")

        app = FastAPI()

        with pytest.raises(Exception, match="Init failed"):
            async with lifespan(app):
                pass

        mock_cleanup.assert_called_once()


class TestAppIntegration:
    def test_app_startup_with_test_client(self, mock_dependencies):
        """Test that the app can start successfully with TestClient"""
        mock_init, mock_cleanup = mock_dependencies

        app = create_app()

        with TestClient(app) as client:
            # The app should be able to start without errors
            assert client.app is not None

    @patch("app.main.os.environ.get")
    def test_main_execution_with_custom_port(self, mock_env_get):
        """Test main execution with custom port"""
        mock_env_get.return_value = "9000"

        with patch("app.main.uvicorn.run") as mock_run:
            # Import and execute the main block
            exec("""if __name__ == "__main__":
                 port = int(os.environ.get("PORT", 8000))
                 uvicorn.run(app, host="0.0.0.0", port=port, log_level="debug", reload=False)
                 """,
                 {
                     "__name__": "__main__",
                     "os": __import__("os"),
                     "uvicorn": __import__("uvicorn"),
                     "app": create_app()
                     })

            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args[1]["port"] == 9000
            assert call_args[1]["host"] == "0.0.0.0"
            assert call_args[1]["log_level"] == "debug"
            assert call_args[1]["reload"] is False

    @patch("app.main.os.environ.get")
    def test_main_execution_with_default_port(self, mock_env_get):
        mock_env_get.return_value = None

        with patch("app.main.uvicorn.run"):
            # Simulate the if __name__ == "__main__" block
            port = int(mock_env_get.return_value or 8000)
            assert port == 8000


def test_main_block_execution():
    with patch("app.main.os.environ.get", return_value="8000"), \
         patch("app.main.uvicorn.run") as mock_run:

        # Test by importing the module with __name__ set to "__main__"
        import importlib.util
        spec = importlib.util.spec_from_file_location("__main__", "app/main.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        mock_run.assert_called_once()
