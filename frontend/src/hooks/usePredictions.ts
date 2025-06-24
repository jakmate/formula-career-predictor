import { useState, useEffect, useCallback, useRef } from 'react';
import type { SystemStatus } from '../types/SystemStatus';
import type { ModelResults } from '../types/ModelResults';

interface AllPredictionsResponse {
  models: string[];
  predictions: Record<string, ModelResults>;
  system_status: SystemStatus;
}

export const usePredictions = () => {
  const [allData, setAllData] = useState<AllPredictionsResponse | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const refreshStatusRef = useRef<SystemStatus | null>(null); // Track original status

  const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  // Fetch without state dependencies
  const fetchPredictions = useCallback(async (): Promise<AllPredictionsResponse> => {
    const response = await fetch(`${API_BASE}/predictions`);
    if (!response.ok) throw new Error('Failed to fetch predictions');
    return response.json();
  }, [API_BASE]);

  const fetchAllPredictions = useCallback(async () => {
    try {
      setLoading(true);
      const data = await fetchPredictions();
      setAllData(data);
      if (data.models.length > 0 && !selectedModel) {
        setSelectedModel(data.models[0]);
      }
      setError(null);
      return data; // Return for immediate use
    } catch (err) {
      setError(`Failed to fetch predictions: ${err instanceof Error ? err.message : 'Unknown error'}`);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [fetchPredictions, selectedModel]);

  const handleRefresh = async () => {
    try {
      setLoading(true);
      // Capture current status before refresh
      refreshStatusRef.current = allData?.system_status || null;
      
      await fetch(`${API_BASE}/refresh`, { method: 'POST' });
      
      const maxAttempts = 10;
      let attempts = 0;
      
      const checkForUpdates = async () => {
        try {
          // Fetch data
          const newData = await fetchPredictions();
          setAllData(newData);
          
          // Compare with original status (using ref)
          const hasNewData = (
            newData.system_status.last_scrape !== refreshStatusRef.current?.last_scrape ||
            newData.system_status.last_training !== refreshStatusRef.current?.last_training
          );
          
          // Success - stop checking
          if (hasNewData) {
            setLoading(false);
            return;
          }
          
          if (++attempts >= maxAttempts) {
            setLoading(false);
            setError('Update check timeout');
            return; // Fail - stop after max attempts
          }
          
          // Continue polling
          setTimeout(checkForUpdates, 3000);
        } catch (err) {
          setLoading(false);
          setError(`Update failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
        }
      };
      
      checkForUpdates();
    } catch (err) {
      setLoading(false);
      setError(`Refresh failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  };

  useEffect(() => {
    fetchAllPredictions();
  }, [fetchAllPredictions]);

  return {
    predictions: allData?.predictions || {},
    selectedModel,
    setSelectedModel,
    models: allData?.models || [],
    loading,
    status: allData?.system_status || null,
    error,
    handleRefresh,
    currentPredictions: allData?.predictions[selectedModel]?.predictions || []
  };
};
