import { useState, useEffect, useCallback } from 'react';
import type { SystemStatus } from '../types/SystemStatus';
import type { ModelResults } from '../types/ModelResults';

export const usePredictions = () => {
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
      setError(`Failed to fetch models: ${err instanceof Error ? err.message : 'Unknown error'}`);
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
      setError(`Failed to trigger update ${err instanceof Error ? err.message : 'Unknown error'}`);
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

  return {
    predictions,
    selectedModel,
    setSelectedModel,
    models,
    loading,
    status,
    error,
    handleRefresh,
    currentPredictions: predictions[selectedModel]?.predictions || []
  };
};
