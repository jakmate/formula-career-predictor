import { useState, useEffect, useCallback, useRef } from 'react';
import type { SystemStatus } from '../types/SystemStatus';
import type { ModelResults } from '../types/ModelResults';

export type SeriesType = 'f3_to_f2' | 'f2_to_f1';

interface PredictionsResponse {
  models: string[];
  predictions: Record<string, ModelResults>;
  system_status: SystemStatus;
}

export const usePredictions = (series: SeriesType = 'f3_to_f2') => {
  const [allData, setAllData] = useState<PredictionsResponse | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const refreshStatusRef = useRef<SystemStatus | null>(null);

  const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  const fetchPredictions =
    useCallback(async (): Promise<PredictionsResponse> => {
      try {
        const response = await fetch(`${API_BASE}/api/predictions/${series}`);
        if (!response.ok) {
          throw new Error('Server responded with an error');
        }
        return response.json();
      } catch {
        throw new Error('Network connection issue');
      }
    }, [API_BASE, series]);

  const fetchAllPredictions = useCallback(async () => {
    try {
      setLoading(true);
      const data = await fetchPredictions();
      setAllData(data);
      setError(null);
      return data;
    } catch (err) {
      setError(
        'Failed to load predictions data. Please check your connection.'
      );
      console.error('Fetch error:', err);
    } finally {
      setLoading(false);
    }
  }, [fetchPredictions]);

  // Set initial model when data loads
  useEffect(() => {
    if (allData?.models?.length && !selectedModel) {
      setSelectedModel(allData.models[0]);
    }
  }, [allData?.models, selectedModel]);

  const handleRefresh = async () => {
    try {
      setLoading(true);
      setError(null);
      refreshStatusRef.current = allData?.system_status || null;

      const refreshResponse = await fetch(`${API_BASE}/api/system/refresh`, {
        method: 'POST',
      });
      if (!refreshResponse.ok) {
        throw new Error('Refresh request failed');
      }

      const maxAttempts = 10;
      let attempts = 0;

      const checkForUpdates = async () => {
        try {
          const newData = await fetchPredictions();
          setAllData(newData);

          const hasNewData =
            newData.system_status?.last_scrape !==
              refreshStatusRef.current?.last_scrape ||
            newData.system_status?.last_training !==
              refreshStatusRef.current?.last_training;

          if (hasNewData) {
            setLoading(false);
            return;
          }

          if (++attempts >= maxAttempts) {
            setLoading(false);
            setError('Update check timeout');
            return;
          }

          setTimeout(checkForUpdates, 3000);
        } catch (err) {
          setLoading(false);
          setError(
            `Update failed: ${err instanceof Error ? err.message : 'Unknown error'}`
          );
          return;
        }
      };

      await checkForUpdates();
    } catch (err) {
      setLoading(false);
      setError('Could not refresh data. Please try again later.');
      console.error('Refresh error:', err);
    }
  };

  // Initial load - only runs once
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
    currentPredictions:
      allData?.predictions?.[selectedModel]?.predictions || [],
  };
};
