import { describe, it, expect } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useSorting } from './useSorting';
import type { Driver } from '../types/Driver';

const mockDrivers: Driver[] = [
  {
    driver: 'Lewis Hamilton',
    nationality: 'British',
    empirical_percentage: 85.5,
    wins: 13,
    podiums: 19,
    points: 440.5,
    position: 1,
    team: 'Mercedes',
    win_rate: 70,
    top_10_rate: 0,
    experience: 0,
    dnf_rate: 0,
    participation_rate: 0,
    team_pos: 0,
    team_points: 0,
    teammate_h2h: 0,
  },
  {
    driver: 'Max Verstappen',
    nationality: 'Dutch',
    empirical_percentage: 92.3,
    wins: 5,
    podiums: 9,
    points: 258.5,
    position: 2,
    team: 'Red Bull',
    win_rate: 54,
    top_10_rate: 0,
    experience: 0,
    dnf_rate: 0,
    participation_rate: 0,
    team_pos: 0,
    team_points: 0,
    teammate_h2h: 0,
  },
  {
    driver: 'Charles Leclerc',
    nationality: 'Monégasque',
    empirical_percentage: 78.2,
    wins: 2,
    podiums: 5,
    points: 103,
    position: 3,
    team: 'Ferrari',
    win_rate: 32,
    top_10_rate: 0,
    experience: 0,
    dnf_rate: 0,
    participation_rate: 0,
    team_pos: 0,
    team_points: 0,
    teammate_h2h: 0,
  },
];

describe('useSorting', () => {
  describe('initialization', () => {
    it('initializes with default field and desc direction', () => {
      const { result } = renderHook(() => useSorting(mockDrivers));

      expect(result.current.sortConfig).toEqual({
        field: 'empirical_percentage',
        direction: 'desc',
      });
    });

    it('initializes with custom default field', () => {
      const { result } = renderHook(() => useSorting(mockDrivers, 'driver'));

      expect(result.current.sortConfig).toEqual({
        field: 'driver',
        direction: 'desc',
      });
    });

    it('sorts data by default field on initialization', () => {
      const { result } = renderHook(() => useSorting(mockDrivers));

      expect(result.current.sortedData[0].driver).toBe('Max Verstappen');
      expect(result.current.sortedData[1].driver).toBe('Lewis Hamilton');
      expect(result.current.sortedData[2].driver).toBe('Charles Leclerc');
    });
  });

  describe('numerical sorting', () => {
    it('sorts numbers in descending order by default', () => {
      const { result } = renderHook(() => useSorting(mockDrivers, 'win_rate'));

      expect(result.current.sortedData.map((d) => d.win_rate)).toEqual([
        70, 54, 32,
      ]);
    });

    it('sorts numbers in ascending order when toggled', () => {
      const { result } = renderHook(() => useSorting(mockDrivers, 'win_rate'));

      act(() => {
        result.current.handleSort('win_rate');
      });

      expect(result.current.sortedData.map((d) => d.win_rate)).toEqual([
        32, 54, 70,
      ]);
    });

    it('handles decimal numbers correctly', () => {
      const { result } = renderHook(() =>
        useSorting(mockDrivers, 'empirical_percentage')
      );

      expect(
        result.current.sortedData.map((d) => d.empirical_percentage)
      ).toEqual([92.3, 85.5, 78.2]);
    });
  });

  describe('string sorting', () => {
    it('sorts strings in descending order by default', () => {
      const { result } = renderHook(() => useSorting(mockDrivers, 'driver'));

      expect(result.current.sortedData.map((d) => d.driver)).toEqual([
        'Max Verstappen',
        'Lewis Hamilton',
        'Charles Leclerc',
      ]);
    });

    it('sorts strings in ascending order when toggled', () => {
      const { result } = renderHook(() => useSorting(mockDrivers, 'driver'));

      act(() => {
        result.current.handleSort('driver');
      });

      expect(result.current.sortedData.map((d) => d.driver)).toEqual([
        'Charles Leclerc',
        'Lewis Hamilton',
        'Max Verstappen',
      ]);
    });

    it('handles case-insensitive string sorting', () => {
      const mixedCaseDrivers = [
        { ...mockDrivers[0], driver: 'alice' },
        { ...mockDrivers[1], driver: 'Bob' },
        { ...mockDrivers[2], driver: 'CHARLIE' },
      ];

      const { result } = renderHook(() =>
        useSorting(mixedCaseDrivers, 'driver')
      );

      act(() => {
        result.current.handleSort('driver');
      });

      expect(result.current.sortedData.map((d) => d.driver)).toEqual([
        'alice',
        'Bob',
        'CHARLIE',
      ]);
    });
  });

  describe('sort direction toggling', () => {
    it('toggles from desc to asc when clicking same field', () => {
      const { result } = renderHook(() => useSorting(mockDrivers, 'win_rate'));

      act(() => {
        result.current.handleSort('win_rate');
      });

      expect(result.current.sortConfig.direction).toBe('asc');
    });

    it('toggles from asc to desc when clicking same field twice', () => {
      const { result } = renderHook(() => useSorting(mockDrivers, 'win_rate'));

      act(() => {
        result.current.handleSort('win_rate');
        result.current.handleSort('win_rate');
      });

      expect(result.current.sortConfig.direction).toBe('desc');
    });

    it('defaults to desc when switching to different field', () => {
      const { result } = renderHook(() => useSorting(mockDrivers, 'win_rate'));

      act(() => {
        result.current.handleSort('win_rate'); // Toggle to asc
        result.current.handleSort('driver'); // Switch field
      });

      expect(result.current.sortConfig).toEqual({
        field: 'driver',
        direction: 'desc',
      });
    });
  });

  describe('edge cases', () => {
    it('handles empty data array', () => {
      const { result } = renderHook(() => useSorting([]));

      expect(result.current.sortedData).toEqual([]);
    });

    it('handles single item array', () => {
      const singleDriver = [mockDrivers[0]];
      const { result } = renderHook(() => useSorting(singleDriver));

      expect(result.current.sortedData).toEqual(singleDriver);
    });

    it('maintains original data immutability', () => {
      const originalData = [...mockDrivers];
      const { result } = renderHook(() => useSorting(mockDrivers));

      expect(mockDrivers).toEqual(originalData);
      expect(result.current.sortedData).not.toBe(mockDrivers);
    });
  });

  describe('data updates', () => {
    it('re-sorts when data changes', () => {
      const { result, rerender } = renderHook(
        ({ data }) => useSorting(data, 'driver'),
        { initialProps: { data: mockDrivers.slice(0, 2) } }
      );

      expect(result.current.sortedData).toHaveLength(2);

      rerender({ data: mockDrivers });

      expect(result.current.sortedData).toHaveLength(3);
      expect(result.current.sortedData.map((d) => d.driver)).toEqual([
        'Max Verstappen',
        'Lewis Hamilton',
        'Charles Leclerc',
      ]);
    });

    it('maintains sort config when data changes', () => {
      const { result, rerender } = renderHook(
        ({ data }) => useSorting(data, 'driver'),
        { initialProps: { data: mockDrivers.slice(0, 2) } }
      );

      act(() => {
        result.current.handleSort('driver'); // Toggle to asc
      });

      expect(result.current.sortConfig.direction).toBe('asc');

      rerender({ data: mockDrivers });

      expect(result.current.sortConfig.direction).toBe('asc');
      expect(result.current.sortedData.map((d) => d.driver)).toEqual([
        'Charles Leclerc',
        'Lewis Hamilton',
        'Max Verstappen',
      ]);
    });
  });
});
