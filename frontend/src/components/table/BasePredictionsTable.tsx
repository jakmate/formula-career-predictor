import {
  Calendar,
  RefreshCw,
  Target,
  TrendingUp,
  UserRound,
} from 'lucide-react';
import { ErrorDisplay } from '../ErrorDisplay';
import { BaseTableContent } from './TableContent';
import { usePredictions, type SeriesType } from '../../hooks/usePredictions';
import { Header } from '../Header';
import { useState } from 'react';

interface SeriesOption {
  value: string;
  label: string;
}

interface BasePredictionsTableProps {
  defaultSeries: SeriesType;
  seriesOptions: SeriesOption[];
  getTitle: (selectedSeries: SeriesType) => string;
  getDescription: (selectedSeries: SeriesType) => string;
}

export const BasePredictionsTable = ({
  defaultSeries,
  seriesOptions,
  getTitle,
  getDescription,
}: BasePredictionsTableProps) => {
  const [selectedSeries, setSelectedSeries] = useState<SeriesType>(
    defaultSeries as SeriesType
  );
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

  return (
    <div className="w-full">
      <Header
        title={getTitle(selectedSeries)}
        description={getDescription(selectedSeries)}
        rightContent={
          <div className="flex flex-col sm:flex-row gap-3">
            <select
              value={selectedSeries}
              onChange={(e) => setSelectedSeries(e.target.value as SeriesType)}
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
                Models: {status.models_available?.[selectedSeries]?.length || 0}
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
          <BaseTableContent predictions={currentPredictions} />
        )}
      </div>
    </div>
  );
};
