import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { act, renderHook, waitFor } from '@testing-library/react';
import { useSchedule } from './useSchedule';

// Mock fetch globally
const mockFetch = vi.fn();
globalThis.fetch = mockFetch;

// Mock environment variable
vi.mock('import.meta', () => ({
  env: {
    VITE_API_URL: 'http://localhost:8000',
  },
}));

describe('useSchedule', () => {
  const mockRacesData = [
    { id: 1, name: 'Bahrain Grand Prix', date: '2024-03-02T15:00:00Z' },
    { id: 2, name: 'Saudi Arabian Grand Prix', date: '2024-03-09T16:00:00Z' },
  ];

  const mockNextRaceData = {
    id: 1,
    name: 'Bahrain Grand Prix',
    date: '2024-03-02T15:00:00Z',
  };

  beforeEach(() => {
    vi.clearAllMocks();
    // Mock successful responses by default
    mockFetch.mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue(mockRacesData),
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should initialize with default values', () => {
    const { result } = renderHook(() => useSchedule());

    expect(result.current.races).toEqual([]);
    expect(result.current.nextRace).toBeNull();
    expect(result.current.selectedSeries).toBe('f1');
    expect(result.current.loading).toBe(true);
    expect(result.current.error).toBeNull();
    expect(result.current.series).toEqual([
      { value: 'f1', label: 'Formula 1' },
      { value: 'f2', label: 'Formula 2' },
      { value: 'f3', label: 'Formula 3' },
    ]);
  });

  it('should fetch schedule data on mount', async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: vi.fn().mockResolvedValue(mockRacesData),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: vi.fn().mockResolvedValue(mockNextRaceData),
      });

    const { result } = renderHook(() => useSchedule());

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(mockFetch).toHaveBeenCalledTimes(2);
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/races/f1?timezone='),
      expect.objectContaining({
        headers: expect.objectContaining({
          'X-Timezone': expect.any(String),
          'Content-Type': 'application/json',
        }),
      })
    );
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/races/f1/next?timezone='),
      expect.objectContaining({
        headers: expect.objectContaining({
          'X-Timezone': expect.any(String),
          'Content-Type': 'application/json',
        }),
      })
    );

    expect(result.current.races).toEqual(mockRacesData);
    expect(result.current.nextRace).toEqual(mockNextRaceData);
    expect(result.current.error).toBeNull();
  });

  it('should handle races API error', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
    });

    const { result } = renderHook(() => useSchedule());

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.error).toBe('Failed to fetch F1 schedule');
    expect(result.current.races).toEqual([]);
  });

  it('should handle next race API error gracefully', async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: vi.fn().mockResolvedValue(mockRacesData),
      })
      .mockResolvedValueOnce({
        ok: false,
        status: 404,
      });

    const { result } = renderHook(() => useSchedule());

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.races).toEqual(mockRacesData);
    expect(result.current.nextRace).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it('should handle network errors', async () => {
    mockFetch.mockRejectedValueOnce(new Error('Network error'));

    const { result } = renderHook(() => useSchedule());

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.error).toBe('Network error');
    expect(result.current.races).toEqual([]);
  });

  it('should handle unknown errors', async () => {
    mockFetch.mockRejectedValueOnce('Unknown error');

    const { result } = renderHook(() => useSchedule());

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.error).toBe('An unknown error occurred');
  });

  it('should include timezone in API calls', async () => {
    const { result } = renderHook(() => useSchedule());

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    const expectedTimezone = Intl.DateTimeFormat().resolvedOptions().timeZone;

    expect(mockFetch).toHaveBeenCalledWith(
      `http://localhost:8000/api/races/f1?timezone=${encodeURIComponent(expectedTimezone)}`,
      expect.objectContaining({
        headers: expect.objectContaining({
          'X-Timezone': expectedTimezone,
        }),
      })
    );
  });

  it('calls fetch again when refetch is invoked', async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: vi.fn().mockResolvedValue(mockRacesData),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: vi.fn().mockResolvedValue(mockNextRaceData),
      });

    const { result } = renderHook(() => useSchedule());

    // wait for initial fetch to finish
    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(mockFetch).toHaveBeenCalledTimes(2);

    // prepare mock for refetch
    mockFetch.mockClear();
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: vi.fn().mockResolvedValue(mockRacesData),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: vi.fn().mockResolvedValue(mockNextRaceData),
      });

    await act(async () => {
      result.current.refetch();
    });

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(mockFetch).toHaveBeenCalledTimes(2);

    expect(result.current.races).toEqual(mockRacesData);
    expect(result.current.nextRace).toEqual(mockNextRaceData);
  });
});
