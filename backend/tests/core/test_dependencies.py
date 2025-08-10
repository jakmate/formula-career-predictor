import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.dependencies import (
    initialize_app_state,
    cleanup_app_state,
    get_app_state,
    get_model_service,
    get_data_service,
    get_scheduler_service
)


@pytest.fixture
def mock_services():
    """Mock all service classes"""
    with patch("app.dependencies.AppState") as mock_app_state, \
         patch("app.dependencies.ModelService") as mock_model_service, \
         patch("app.dependencies.DataService") as mock_data_service, \
         patch("app.dependencies.SchedulerService") as mock_scheduler_service:

        # Configure mocks
        mock_app_state_instance = MagicMock()
        mock_app_state.return_value = mock_app_state_instance

        mock_model_service_instance = AsyncMock()
        mock_model_service.return_value = mock_model_service_instance

        mock_data_service_instance = AsyncMock()
        mock_data_service.return_value = mock_data_service_instance

        mock_scheduler_service_instance = AsyncMock()
        mock_scheduler_service.return_value = mock_scheduler_service_instance

        yield {
            "app_state": mock_app_state,
            "model_service": mock_model_service,
            "data_service": mock_data_service,
            "scheduler_service": mock_scheduler_service,
            "app_state_instance": mock_app_state_instance,
            "model_service_instance": mock_model_service_instance,
            "data_service_instance": mock_data_service_instance,
            "scheduler_service_instance": mock_scheduler_service_instance
        }


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state before each test"""
    import app.dependencies
    app.dependencies.app_state = None
    app.dependencies.model_service = None
    app.dependencies.data_service = None
    app.dependencies.scheduler_service = None
    yield
    # Reset after test
    app.dependencies.app_state = None
    app.dependencies.model_service = None
    app.dependencies.data_service = None
    app.dependencies.scheduler_service = None


class TestInitializeAppState:
    @pytest.mark.asyncio
    async def test_initialize_app_state_with_existing_models(self, mock_services):
        mocks = mock_services
        mocks["model_service_instance"].load_models.return_value = True

        await initialize_app_state()

        # Verify services were created
        mocks["app_state"].assert_called_once()
        mocks["app_state_instance"].load_state.assert_called_once()
        mocks["model_service"].assert_called_once()
        mocks["data_service"].assert_called_once()
        mocks["scheduler_service"].assert_called_once()

        # Verify model loading was attempted
        mocks["model_service_instance"].load_models.assert_called_once()

        # Verify data service initialization was NOT called
        mocks["data_service_instance"].initialize_system.assert_not_called()

        # Verify scheduler was started
        mocks["scheduler_service_instance"].start.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_app_state_without_models(self, mock_services):
        mocks = mock_services
        mocks["model_service_instance"].load_models.return_value = False

        with patch("app.dependencies.LOGGER") as mock_logger:
            await initialize_app_state()

        # Verify model loading was attempted
        mocks["model_service_instance"].load_models.assert_called_once()

        # Verify data service initialization was called
        mocks["data_service_instance"].initialize_system.assert_called_once()

        # Verify logger was called
        mock_logger.info.assert_called_with("No models found. Initializing system...")

        # Verify scheduler was started
        mocks["scheduler_service_instance"].start.assert_called_once()


class TestCleanupAppState:
    @pytest.mark.asyncio
    async def test_cleanup_with_services(self, mock_services):
        # Initialize first
        await initialize_app_state()

        await cleanup_app_state()

        # Verify scheduler was stopped
        mock_services["scheduler_service_instance"].stop.assert_called_once()

        # Verify state was saved
        mock_services["app_state_instance"].save_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_without_services(self):
        # Don't initialize - services should be None
        await cleanup_app_state()

        # Should not raise any errors


class TestGetDependencies:
    def test_get_app_state_success(self, mock_services):
        # Set up global state
        import app.dependencies
        app.dependencies.app_state = mock_services["app_state_instance"]

        result = get_app_state()
        assert result == mock_services["app_state_instance"]

    def test_get_app_state_not_initialized(self):
        with pytest.raises(RuntimeError, match="Application state not initialized"):
            get_app_state()

    def test_get_model_service_success(self, mock_services):
        import app.dependencies
        app.dependencies.model_service = mock_services["model_service_instance"]

        result = get_model_service()
        assert result == mock_services["model_service_instance"]

    def test_get_model_service_not_initialized(self):
        with pytest.raises(RuntimeError, match="Model service not initialized"):
            get_model_service()

    def test_get_data_service_success(self, mock_services):
        import app.dependencies
        app.dependencies.data_service = mock_services["data_service_instance"]

        result = get_data_service()
        assert result == mock_services["data_service_instance"]

    def test_get_data_service_not_initialized(self):
        with pytest.raises(RuntimeError, match="Data service not initialized"):
            get_data_service()

    def test_get_scheduler_service_success(self, mock_services):
        import app.dependencies
        app.dependencies.scheduler_service = mock_services["scheduler_service_instance"]

        result = get_scheduler_service()
        assert result == mock_services["scheduler_service_instance"]

    def test_get_scheduler_service_not_initialized(self):
        with pytest.raises(RuntimeError, match="Scheduler service not initialized"):
            get_scheduler_service()


class TestFullLifecycle:
    @pytest.mark.asyncio
    async def test_full_lifecycle(self, mock_services):
        """Test complete initialization and cleanup cycle"""
        mocks = mock_services
        mocks["model_service_instance"].load_models.return_value = True

        # Initialize
        await initialize_app_state()

        # Verify all dependencies are available
        assert get_app_state() == mocks["app_state_instance"]
        assert get_model_service() == mocks["model_service_instance"]
        assert get_data_service() == mocks["data_service_instance"]
        assert get_scheduler_service() == mocks["scheduler_service_instance"]

        # Cleanup
        await cleanup_app_state()

        # Verify cleanup was called
        mocks["scheduler_service_instance"].stop.assert_called_once()
        mocks["app_state_instance"].save_state.assert_called_once()
