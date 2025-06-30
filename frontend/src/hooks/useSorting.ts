import { useState, useMemo } from 'react';
import type { Driver } from '../types/Driver';
import type { SortField, SortConfig } from '../types/Sorting';

export const useSorting = (
  data: Driver[],
  defaultField: SortField = 'empirical_percentage'
) => {
  const [sortConfig, setSortConfig] = useState<SortConfig>({
    field: defaultField,
    direction: 'desc',
  });

  const handleSort = (field: SortField) => {
    setSortConfig((prev) => ({
      field,
      direction:
        prev.field === field && prev.direction === 'desc' ? 'asc' : 'desc',
    }));
  };

  const sortedData = useMemo(() => {
    if (!data.length) return [];

    return [...data].sort((a: Driver, b: Driver) => {
      const aValue = a[sortConfig.field];
      const bValue = b[sortConfig.field];

      let comparison = 0;

      if (typeof aValue === 'string' && typeof bValue === 'string') {
        comparison = aValue.toLowerCase().localeCompare(bValue.toLowerCase());
      } else if (typeof aValue === 'number' && typeof bValue === 'number') {
        comparison = aValue - bValue;
      }

      return sortConfig.direction === 'asc' ? comparison : -comparison;
    });
  }, [data, sortConfig]);

  return { sortedData, sortConfig, handleSort };
};
