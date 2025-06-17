import { RefreshCw } from 'lucide-react';
import { ErrorDisplay } from '../ErrorDisplay';
import { Header } from '../Header';
import { ModelInfo } from '../ModelInfo';
import { TableContent } from '../table/TableContent';
import { usePredictions } from '../../hooks/usePredictions';

const PredictionsTable = () => {
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
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 p-4">
      <div className="max-w-7xl mx-auto">
        <Header 
          selectedModel={selectedModel}
          setSelectedModel={setSelectedModel}
          models={models}
          loading={loading}
          status={status}
          onRefresh={handleRefresh}
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

        {predictions[selectedModel] &&
          <ModelInfo 
            selectedModel={selectedModel}
            predictionCount={predictions[selectedModel].predictions.length}
          />}
      </div>
    </div>
  );
};

export default PredictionsTable