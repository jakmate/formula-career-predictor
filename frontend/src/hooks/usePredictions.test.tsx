import { renderHook, waitFor, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { usePredictions } from './usePredictions';

// Mock fetch
const mockFetch = vi.fn();
globalThis.fetch = mockFetch;

// Mock console.error
const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

describe('usePredictions', () => {
  const mockApiResponse = {
    models: ['model1', 'model2'],
    predictions: {
      model1: { predictions: [{ id: 1 }] },
      model2: { predictions: [{ id: 2 }] },
    },
    system_status: {
      last_scrape_predictions: '2023-01-01T00:00:00Z',
      last_training: '2023-01-02T00:00:00Z',
    },
  };

  beforeEach(() => {
    mockFetch.mockClear();
    consoleSpy.mockClear();
  });

  it('loads predictions on mount', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockApiResponse),
    });

    const { result } = renderHook(() => usePredictions('f3_to_f2'));

    expect(result.current.loading).toBe(true);

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.predictions).toEqual(mockApiResponse.predictions);
    expect(result.current.models).toEqual(['model1', 'model2']);
    expect(result.current.selectedModel).toBe('model1');
    expect(result.current.status).toEqual(mockApiResponse.system_status);
    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:8000/api/predictions/f3_to_f2'
    );
  });

  it('handles fetch error with server error response', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
    });

    const { result } = renderHook(() => usePredictions());

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.error).toBe(
      'Failed to load predictions data. Please check your connection.'
    );
    expect(consoleSpy).toHaveBeenCalled();
  });

  it('handles network error', async () => {
    mockFetch.mockRejectedValueOnce(new Error('Network error'));

    const { result } = renderHook(() => usePredictions());

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.error).toBe(
      'Failed to load predictions data. Please check your connection.'
    );
  });

  it('updates selected model', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockApiResponse),
    });

    const { result } = renderHook(() => usePredictions());

    await waitFor(() => {
      expect(result.current.selectedModel).toBe('model1');
    });

    act(() => {
      result.current.setSelectedModel('model2');
    });

    expect(result.current.selectedModel).toBe('model2');
    expect(result.current.currentPredictions).toEqual([{ id: 2 }]);
  });

  it('handles refresh request failure', async () => {
    // Initial load
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockApiResponse),
    });

    const { result } = renderHook(() => usePredictions());

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    // Refresh request fails
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
    });

    await act(async () => {
      result.current.refreshPredictions();
    });

    await waitFor(() => {
      expect(result.current.error).toBe(
        'Could not refresh data. Please try again later.'
      );
    });
  });

  it('uses environment API URL', async () => {
    const originalEnv = import.meta.env.VITE_API_URL;
    import.meta.env.VITE_API_URL = 'https://api.example.com';

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockApiResponse),
    });

    renderHook(() => usePredictions());

    expect(mockFetch).toHaveBeenCalledWith(
      'https://api.example.com/api/predictions/f3_to_f2'
    );

    import.meta.env.VITE_API_URL = originalEnv;
  });

  it('handles successful refresh with updated data', async () => {
    // Mock environment variable
    const originalEnv = import.meta.env.VITE_API_URL;
    import.meta.env.VITE_API_URL = 'http://localhost:8000';
    // Initial load
    const initialResponse = {
      models: ['model1'],
      predictions: { model1: { predictions: [{ id: 1 }] } },
      system_status: {
        last_scrape_predictions: '2023-01-01T00:00:00Z',
        last_training: '2023-01-02T00:00:00Z',
      },
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(initialResponse),
    });

    const { result } = renderHook(() => usePredictions());

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    // Mock refresh POST request
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({}),
    });

    // Mock updated data response
    const updatedResponse = {
      ...initialResponse,
      system_status: {
        last_scrape_predictions: '2023-01-01T01:00:00Z', // Updated timestamp
        last_training: '2023-01-02T00:00:00Z',
      },
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(updatedResponse),
    });

    await act(async () => {
      result.current.refreshPredictions();
    });

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.status?.last_scrape_predictions).toBe(
      '2023-01-01T01:00:00Z'
    );
    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:8000/api/system/refresh/predictions',
      { method: 'POST' }
    );

    // Restore environment
    import.meta.env.VITE_API_URL = originalEnv;
  });

  it('handles refresh timeout after max attempts', async () => {
    // Mock environment variable
    const originalEnv = import.meta.env.VITE_API_URL;
    import.meta.env.VITE_API_URL = 'http://localhost:8000';

    // Initial load
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockApiResponse),
    });

    const { result } = renderHook(() => usePredictions());

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    // Mock refresh POST request
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({}),
    });

    // Mock 11 polling responses that never change (triggers timeout)
    for (let i = 0; i < 11; i++) {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockApiResponse), // Same timestamps
      });
    }

    await act(async () => {
      result.current.refreshPredictions();
    });

    await waitFor(
      () => {
        expect(result.current.error).toBe('Update check timeout');
        expect(result.current.loading).toBe(false);
      },
      { timeout: 35000 } // Allow time for all polling attempts
    );

    // Restore environment
    import.meta.env.VITE_API_URL = originalEnv;
  }, 40000);
});
