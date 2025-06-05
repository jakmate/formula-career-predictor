import { useState, useEffect, useCallback } from 'react';
import { RefreshCw, Calendar, TrendingUp, Trophy, Target } from 'lucide-react';

interface Driver {
  driver: string;
  position: number;
  points: number;
  win_rate: number;
  podium_rate: number;
  top_10_rate: number;
  experience: number;
  empirical_percentage: number;
  prediction: number;
}

interface ModelResults {
  model_name: string;
  predictions: Driver[];
  accuracy_metrics: { total_predictions: number };
}

interface SystemStatus {
  last_scrape?: string;
  last_training?: string;
  models_available?: string[];
}

const PredictionsTable = () => {
  const [predictions, setPredictions] = useState<Record<string, ModelResults>>({});
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [models, setModels] = useState<string[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  const fetchModels = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/models`);
      const data = await response.json();
      const allModels = [...data.ml_models, ...data.deep_learning_models];
      setModels(allModels);
      if (allModels.length > 0 && !selectedModel) {
        setSelectedModel(allModels[0]);
      }
    } catch (err) {
      setError('Failed to fetch models');
    }
  }, [API_BASE, selectedModel]);

  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/status`);
      const data = await response.json();
      setStatus(data);
    } catch (err) {
      console.error('Error fetching status:', err);
    }
  }, [API_BASE]);

  const fetchPredictions = useCallback(async (modelName: string = selectedModel) => {
    if (!modelName) return;
    
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE}/predictions/${modelName}`);
      if (!response.ok) throw new Error('Failed to fetch predictions');
      
      const data = await response.json();
      setPredictions(prev => ({ ...prev, [modelName]: data }));
      setError(null);
    } catch (err) {
      setError(`Failed to fetch predictions: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  }, [API_BASE, selectedModel]);

  const handleRefresh = async () => {
    setLoading(true);
    try {
      await fetch(`${API_BASE}/scrape`, { method: 'POST' });
      await fetch(`${API_BASE}/train`, { method: 'POST' });
      setTimeout(() => {
        fetchPredictions();
        fetchStatus();
      }, 5000);
    } catch (err) {
      setError('Failed to trigger update');
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModels();
    fetchStatus();
  }, [fetchModels, fetchStatus]);

  useEffect(() => {
    if (selectedModel) {
      fetchPredictions(selectedModel);
    }
  }, [selectedModel, fetchPredictions]);

  const currentPredictions = predictions[selectedModel]?.predictions || [];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 mb-6 border border-white/20">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
            <div>
              <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-2">
                <Trophy className="text-yellow-400" />
                F3 to F2 Progression Predictions
              </h1>
              <p className="text-blue-200">AI-powered analysis of Formula 3 drivers likely to advance to Formula 2</p>
            </div>
            
            <div className="flex flex-col sm:flex-row gap-3">
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
            </div>
          </div>
          
          {status && (
            <div className="mt-4 flex flex-wrap gap-4 text-sm text-blue-200">
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
          )}
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-500/20 border border-red-500/50 rounded-lg p-4 mb-6 text-red-200">
            {error}
          </div>
        )}

        {/* Predictions Table */}
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
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-white/5">
                  <tr className="text-left text-white/90">
                    <th className="p-4 font-semibold">Driver</th>
                    <th className="p-4 font-semibold">Position</th>
                    <th className="p-4 font-semibold">Points</th>
                    <th className="p-4 font-semibold">Win %</th>
                    <th className="p-4 font-semibold">Podium %</th>
                    <th className="p-4 font-semibold">Top 10 %</th>
                    <th className="p-4 font-semibold">Experience</th>
                    <th className="p-4 font-semibold">F2 Probability</th>
                    <th className="p-4 font-semibold">Prediction</th>
                  </tr>
                </thead>
                <tbody>
                  {currentPredictions.map((driver: Driver) => (
                    <tr 
                      key={driver.driver} 
                      className={`border-t border-white/10 hover:bg-white/5 transition-colors ${
                        driver.prediction === 1 ? 'bg-green-500/10' : ''
                      }`}
                    >
                      <td className="p-4 text-white font-medium">{driver.driver}</td>
                      <td className="p-4 text-white">{driver.position}</td>
                      <td className="p-4 text-white">{driver.points.toFixed(1)}</td>
                      <td className="p-4 text-white">{(driver.win_rate * 100).toFixed(1)}%</td>
                      <td className="p-4 text-white">{(driver.podium_rate * 100).toFixed(1)}%</td>
                      <td className="p-4 text-white">{(driver.top_10_rate * 100).toFixed(1)}%</td>
                      <td className="p-4 text-white">{driver.experience} years</td>
                      <td className="p-4">
                        <div className="flex items-center gap-2">
                          <div className="w-20 h-2 bg-white/20 rounded-full overflow-hidden">
                            <div 
                              className={`h-full transition-all duration-500 ${
                                driver.empirical_percentage > 70 ? 'bg-green-400' :
                                driver.empirical_percentage > 40 ? 'bg-yellow-400' : 'bg-red-400'
                              }`}
                              style={{ width: `${Math.min(driver.empirical_percentage, 100)}%` }}
                            />
                          </div>
                          <span className="text-white text-sm font-medium">
                            {driver.empirical_percentage.toFixed(1)}%
                          </span>
                        </div>
                      </td>
                      <td className="p-4">
                        <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                          driver.prediction === 1 
                            ? 'bg-green-500/20 text-green-300 border border-green-500/30' 
                            : 'bg-red-500/20 text-red-300 border border-red-500/30'
                        }`}>
                          {driver.prediction === 1 ? 'Likely F2' : 'Unlikely F2'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Model Info */}
        {predictions[selectedModel] && (
          <div className="mt-6 bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/20">
            <h3 className="text-white font-semibold mb-2">Model: {selectedModel}</h3>
            <p className="text-blue-200 text-sm">
              Total predictions: {predictions[selectedModel].predictions.length} drivers
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default PredictionsTable;