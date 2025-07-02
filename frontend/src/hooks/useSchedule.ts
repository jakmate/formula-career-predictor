import { useState, useEffect, useCallback } from 'react';

export const useSchedule = () => {
  const [races, setRaces] = useState([]);
  const [nextRace, setNextRace] = useState(null);
  const [selectedSeries, setSelectedSeries] = useState('f1');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
  const userTimezone = Intl.DateTimeFormat().resolvedOptions().timeZone;

  const series = [
    { value: 'f1', label: 'Formula 1' },
    { value: 'f2', label: 'Formula 2' },
    { value: 'f3', label: 'Formula 3' },
  ];

  const fetchSchedule = useCallback(
    async (series: string) => {
      setLoading(true);
      setError(null);

      try {
        const headers = {
          'X-Timezone': userTimezone,
          'Content-Type': 'application/json',
        };

        const [racesResponse, nextRaceResponse] = await Promise.all([
          fetch(
            `${API_BASE}/api/races/${series}?timezone=${encodeURIComponent(userTimezone)}`,
            {
              headers,
            }
          ),
          fetch(
            `${API_BASE}/api/races/${series}/next?timezone=${encodeURIComponent(userTimezone)}`,
            {
              headers,
            }
          ),
        ]);

        if (!racesResponse.ok) {
          throw new Error(`Failed to fetch ${series.toUpperCase()} schedule`);
        }

        if (!nextRaceResponse.ok) {
          console.warn(`No next race found for ${series.toUpperCase()}`);
        }

        const racesData = await racesResponse.json();
        const nextRaceData = nextRaceResponse.ok
          ? await nextRaceResponse.json()
          : null;

        setRaces(racesData);
        setNextRace(nextRaceData);
      } catch (err) {
        if (err instanceof Error) {
          setError(err.message);
        } else {
          setError('An unknown error occurred');
        }
      } finally {
        setLoading(false);
      }
    },
    [API_BASE, userTimezone]
  );

  // Fetch when selectedSeries changes
  useEffect(() => {
    fetchSchedule(selectedSeries);
  }, [selectedSeries, fetchSchedule]);

  const refetch = useCallback(() => {
    fetchSchedule(selectedSeries);
  }, [fetchSchedule, selectedSeries]);

  return {
    races,
    nextRace,
    selectedSeries,
    setSelectedSeries,
    series,
    loading,
    error,
    refetch,
    userTimezone,
  };
};
