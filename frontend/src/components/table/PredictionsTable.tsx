import { Calendar, RefreshCw, Target, TrendingUp } from 'lucide-react';
import { ErrorDisplay } from '../ErrorDisplay';
import { ModelInfo } from '../ModelInfo';
import { TableContent } from '../table/TableContent';
import { usePredictions } from '../../hooks/usePredictions';
import { Header } from '../Header';

export const PredictionsTable = () => {
  const {
    predictions,
    selectedModel,
    setSelectedModel,
    models,
    loading,
    status,
    error,
    handleRefresh,
    currentPredictions
  } = usePredictions();

  return (
    <div className="w-full">
      <Header 
        title="F3 to F2 Predictions"
        description="AI-powered analysis of Formula 3 drivers likely to advance to Formula 2"
        rightContent={
          <>
            <select 
              value={selectedModel} 
              onChange={(e) => setSelectedModel(e.target.value)}
              className="px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white backdrop-blur-sm focus:outline-none focus:border-blue-400"
            >
              <option value="">Select Model</option>
              {models.map(model => (
                <option key={model} value={model} className="text-gray-800">{model}</option>
              ))}
            </select>
            
            <button 
              onClick={handleRefresh}
              disabled={loading}
              className="px-6 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded-lg font-medium transition-colors flex items-center gap-2"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
              {loading ? 'Updating...' : 'Refresh Data'}
            </button>
          </>
        }
        bottomContent={
          status && (
            <div className="flex flex-wrap gap-4 text-sm text-blue-200">
              {status.last_scrape && (
                <div className="flex items-center gap-1">
                  <Calendar className="w-4 h-4" />
                  Last scrape: {new Date(status.last_scrape).toLocaleString()}
                </div>
              )}
              {status.last_training && (
                <div className="flex items-center gap-1">
                  <TrendingUp className="w-4 h-4" />
                  Last training: {new Date(status.last_training).toLocaleString()}
                </div>
              )}
              <div className="flex items-center gap-1">
                <Target className="w-4 h-4" />
                Models: {status.models_available?.length || 0}
              </div>
            </div>
          )
        }
      />

      {error && <ErrorDisplay error={error} />}

      <div className="bg-white/10 backdrop-blur-lg rounded-xl border border-white/20 overflow-hidden">
        {loading && currentPredictions.length === 0 ? (
          <div className="p-12 text-center text-white">
            <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-4 text-blue-400" />
            <p>Loading predictions...</p>
          </div>
        ) : currentPredictions.length === 0 ? (
          <div className="p-12 text-center text-white">
            <p>No predictions available. Select a model and refresh data.</p>
          </div>
        ) : (
          <TableContent predictions={currentPredictions} />
        )}
      </div>

      {predictions[selectedModel] && (
        <ModelInfo 
          selectedModel={selectedModel}
          predictionCount={predictions[selectedModel].predictions.length}
        />
      )}
    </div>
  );
};