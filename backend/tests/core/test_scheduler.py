import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from app.core.scheduler import SchedulerService
from app.core.state import AppState


@pytest.fixture
def mock_app_state():
    app_state = Mock(spec=AppState)
    app_state.system_status = {
        "last_scrape": None,
        "last_trained_season": 2022
    }
    app_state.save_state = Mock()
    return app_state


@pytest.fixture
def mock_services():
    model_service = Mock()
    data_service = Mock()
    data_service.initialize_system = AsyncMock()
    return model_service, data_service


@pytest.fixture
def scheduler_service(mock_app_state, mock_services):
    model_service, data_service = mock_services
    return SchedulerService(mock_app_state, model_service, data_service)


class TestSchedulerService:
    def test_init(self, mock_app_state, mock_services):
        """Test scheduler initialization"""
        model_service, data_service = mock_services
        scheduler = SchedulerService(mock_app_state, model_service, data_service)

        assert scheduler.app_state == mock_app_state
        assert scheduler.model_service == model_service
        assert scheduler.data_service == data_service
        assert scheduler.scheduler is not None

    @pytest.mark.asyncio
    @patch('app.core.scheduler.LOGGER')
    async def test_start(self, mock_logger, scheduler_service):
        """Test scheduler start"""
        with patch.object(scheduler_service.scheduler, 'add_job') as mock_add_job, \
             patch.object(scheduler_service.scheduler, 'start') as mock_start:

            await scheduler_service.start()

            mock_add_job.assert_called_once()
            mock_start.assert_called_once()
            mock_logger.info.assert_called_with("Scheduler started")

    @pytest.mark.asyncio
    async def test_stop(self, scheduler_service):
        """Test scheduler stop"""
        with patch.object(scheduler_service.scheduler, 'shutdown') as mock_shutdown:
            await scheduler_service.stop()
            mock_shutdown.assert_called_once()

    @pytest.mark.asyncio
    @patch('app.core.scheduler.scrape_current_year')
    @patch('app.core.scheduler.LOGGER')
    @patch('app.core.scheduler.datetime')
    async def test_scrape_and_train_task_no_training(self, mock_datetime, mock_logger,
                                                     mock_scrape, scheduler_service):
        """Test scrape and train task when no training needed"""
        # Mock datetime.now()
        mock_now = datetime(2023, 6, 15)
        mock_datetime.now.return_value = mock_now

        # Mock season not complete
        with patch.object(scheduler_service, '_is_season_complete', return_value=False):
            await scheduler_service.scrape_and_train_task()

        # Verify scraping happened
        mock_scrape.assert_called_once()
        scheduler_service.app_state.save_state.assert_called_once()
        assert scheduler_service.app_state.system_status["last_scrape"] == mock_now

    @pytest.mark.asyncio
    @patch('app.core.scheduler.scrape_current_year')
    @patch('app.core.scheduler.LOGGER')
    @patch('app.core.scheduler.CURRENT_YEAR', 2024)
    async def test_scrape_and_train_task_with_training(self, mock_logger, mock_scrape,
                                                       scheduler_service):
        """Test scrape and train task when training needed"""
        # Mock season complete and new season available
        with patch.object(scheduler_service, '_is_season_complete', return_value=True), \
             patch.object(scheduler_service, '_train_models_task', new_callable=AsyncMock) as mock_train: # noqa: 501

            await scheduler_service.scrape_and_train_task()

        mock_scrape.assert_called_once()
        mock_train.assert_called_once()
        scheduler_service.app_state.save_state.assert_called_once()

    @pytest.mark.asyncio
    @patch('app.core.scheduler.scrape_current_year')
    @patch('app.core.scheduler.LOGGER')
    @patch('app.services.prediction_service.PredictionService')
    async def test_scrape_and_train_task_update_predictions(self, mock_prediction_service_class,
                                                            mock_logger, mock_scrape,
                                                            scheduler_service):
        """Test prediction updates when no training needed"""
        # Mock prediction service
        mock_prediction_service = Mock()
        mock_prediction_service.update_predictions = AsyncMock()
        mock_prediction_service_class.return_value = mock_prediction_service

        with patch.object(scheduler_service, '_is_season_complete', return_value=False):
            await scheduler_service.scrape_and_train_task()

        # Verify prediction service was called for each series
        assert mock_prediction_service_class.call_count == 2
        assert mock_prediction_service.update_predictions.call_count == 2

    @pytest.mark.asyncio
    @patch('app.core.scheduler.scrape_current_year')
    @patch('app.core.scheduler.LOGGER')
    async def test_scrape_and_train_task_exception_handling(self, mock_logger, mock_scrape,
                                                            scheduler_service):
        """Test exception handling in scrape and train task"""
        # Make scraping raise exception
        mock_scrape.side_effect = Exception("Scraping failed")

        await scheduler_service.scrape_and_train_task()

        # Verify error logged and state saved
        mock_logger.error.assert_called()
        scheduler_service.app_state.save_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_train_models_task_success(self, scheduler_service):
        """Test successful model training task"""
        await scheduler_service._train_models_task()

        scheduler_service.data_service.initialize_system.assert_called_once()

    @pytest.mark.asyncio
    @patch('app.core.scheduler.LOGGER')
    async def test_train_models_task_exception(self, mock_logger, scheduler_service):
        """Test model training task with exception"""
        scheduler_service.data_service.initialize_system.side_effect = Exception("Training failed")

        await scheduler_service._train_models_task()

        mock_logger.error.assert_called_with("Training task failed: Training failed")

    @patch('app.core.scheduler.CURRENT_YEAR', 2023)
    @patch('app.core.scheduler.SEASON_END_MONTH', 11)
    def test_is_season_complete_true(self, scheduler_service):
        """Test season complete check returns True"""
        with patch('app.core.scheduler.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 12, 15)

            result = scheduler_service._is_season_complete()
            assert result is True

    @patch('app.core.scheduler.CURRENT_YEAR', 2023)
    @patch('app.core.scheduler.SEASON_END_MONTH', 11)
    def test_is_season_complete_false_early_month(self, scheduler_service):
        """Test season complete check returns False (early in season)"""
        with patch('app.core.scheduler.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 9, 15)

            result = scheduler_service._is_season_complete()
            assert result is False

    @patch('app.core.scheduler.CURRENT_YEAR', 2023)
    @patch('app.core.scheduler.SEASON_END_MONTH', 11)
    def test_is_season_complete_false_wrong_year(self, scheduler_service):
        """Test season complete check returns False (wrong year)"""
        with patch('app.core.scheduler.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 12, 15)

            result = scheduler_service._is_season_complete()
            assert result is False

    @pytest.mark.asyncio
    @patch('app.core.scheduler.asyncio.get_event_loop')
    async def test_executor_usage(self, mock_get_loop, scheduler_service):
        """Test that scraping runs in executor"""
        mock_loop = Mock()
        mock_get_loop.return_value = mock_loop
        mock_loop.run_in_executor = AsyncMock()

        with patch.object(scheduler_service, '_is_season_complete', return_value=False), \
             patch('app.services.prediction_service.PredictionService') as mock_ps:

            mock_prediction_service = Mock()
            mock_prediction_service.update_predictions = AsyncMock()
            mock_ps.return_value = mock_prediction_service

            await scheduler_service.scrape_and_train_task()

        mock_loop.run_in_executor.assert_called_once()
