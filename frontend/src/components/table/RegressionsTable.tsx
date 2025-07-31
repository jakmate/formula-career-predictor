import {
  Calendar,
  RefreshCw,
  Target,
  TrendingUp,
  UserRound,
} from 'lucide-react';
import { ErrorDisplay } from '../ErrorDisplay';
import { usePredictions } from '../../hooks/usePredictions';
import { Header } from '../Header';
import { useState } from 'react';
import { RegressionTableContent } from './RegressionTableContext';

export const RegressionTable = () => {
  const [selectedSeries, setSelectedSeries] = useState('f3_regression');
  const {
    predictions,
    selectedModel,
    setSelectedModel,
    models,
    loading,
    status,
    error,
    handleRefresh,
    currentPredictions,
  } = usePredictions(selectedSeries);

  const seriesOptions = [
    { value: 'f3_regression', label: 'F3 Position Predictions' },
    { value: 'f2_regression', label: 'F2 Position Predictions' },
    { value: 'f1_regression', label: 'F1 Position Predictions' },
  ];

  const getTitle = () => {
    switch (selectedSeries) {
      case 'f3_regression':
        return 'F3 Position Predictions';
      case 'f2_regression':
        return 'F2 Position Predictions';
      case 'f1_regression':
        return 'F1 Position Predictions';
      default:
        return 'Position Predictions';
    }
  };

  const getDescription = () => {
    switch (selectedSeries) {
      case 'f3_regression':
        return 'AI-powered predictions of final championship positions for Formula 3 drivers';
      case 'f2_regression':
        return 'AI-powered predictions of final championship positions for Formula 2 drivers';
      case 'f1_regression':
        return 'AI-powered predictions of final championship positions for Formula 1 drivers';
      default:
        return 'AI-powered position predictions';
    }
  };

  return (
    <div className="w-full">
      <Header
        title={getTitle()}
        description={getDescription()}
        rightContent={
          <div className="flex flex-col sm:flex-row gap-3">
            <select
              value={selectedSeries}
              onChange={(e) => setSelectedSeries(e.target.value)}
              className="px-4 py-2 bg-gray-800/60 border border-cyan-500/30 rounded-lg text-white backdrop-blur-sm focus:outline-none focus:ring-1 focus:ring-cyan-500 shadow-sm"
            >
              {seriesOptions.map((option) => (
                <option
                  key={option.value}
                  value={option.value}
                  className="text-white bg-gray-900"
                >
                  {option.label}
                </option>
              ))}
            </select>

            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="px-4 py-2 bg-gray-800/60 border border-cyan-500/30 rounded-lg text-white backdrop-blur-sm focus:outline-none focus:ring-1 focus:ring-cyan-500 shadow-sm"
            >
              <option value="">Select Model</option>
              {models.map((model) => (
                <option
                  key={model}
                  value={model}
                  className="text-white bg-gray-900"
                >
                  {model}
                </option>
              ))}
            </select>

            <button
              onClick={handleRefresh}
              disabled={loading}
              className="px-6 py-2 bg-gradient-to-r from-cyan-600 to-purple-600 hover:from-cyan-500 hover:to-purple-500 disabled:opacity-50 text-white rounded-lg font-medium transition-all duration-200 flex items-center gap-2 shadow-lg shadow-cyan-500/20 hover:shadow-cyan-500/30"
            >
              <RefreshCw
                className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`}
              />
              {loading ? 'Updating...' : 'Refresh'}
            </button>
          </div>
        }
        bottomContent={
          status && (
            <div className="flex flex-wrap gap-4 text-sm text-cyan-300">
              {status.last_scrape && (
                <div className="flex items-center gap-1">
                  <Calendar className="w-4 h-4" />
                  Last scrape: {new Date(status.last_scrape).toLocaleString()}
                </div>
              )}
              {status.last_training && (
                <div className="flex items-center gap-1">
                  <TrendingUp className="w-4 h-4" />
                  Last training:{' '}
                  {new Date(status.last_training).toLocaleString()}
                </div>
              )}
              <div className="flex items-center gap-1">
                <Target className="w-4 h-4" />
                Models: {status.models_available?.length || 0}
              </div>
              <div className="flex items-center gap-1">
                <UserRound className="w-4 h-4" />
                Drivers: {predictions[selectedModel]?.predictions?.length || 0}
              </div>
            </div>
          )
        }
      />

      {error && <ErrorDisplay error={error} />}

      <div className="bg-gray-800/40 backdrop-blur-lg rounded-xl border border-cyan-500/30 overflow-hidden shadow-lg shadow-cyan-500/10">
        {loading && currentPredictions.length === 0 ? (
          <div className="p-12 text-center text-white">
            <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-4 text-cyan-400" />
            <p>Loading predictions...</p>
          </div>
        ) : currentPredictions.length === 0 ? (
          <div className="p-12 text-center text-white">
            <p>No predictions available. Select a model and refresh data.</p>
          </div>
        ) : (
          <RegressionTableContent predictions={currentPredictions} />
        )}
      </div>
    </div>
  );
};
