import { ChevronUp, ChevronDown } from 'lucide-react';
import type { SortConfig, SortField } from '../../types/Sorting';

interface TableHeaderProps {
  field: SortField;
  sortConfig: SortConfig;
  onSort: (field: SortField) => void;
  children: React.ReactNode;
}

export const TableHeader = ({ field, sortConfig, onSort, children }: TableHeaderProps) => {
  const isActive = sortConfig.field === field;
  const SortIcon = isActive ? (
    sortConfig.direction === 'asc' ? 
      <ChevronUp className="w-4 h-4 inline ml-1" /> : 
      <ChevronDown className="w-4 h-4 inline ml-1" />
  ) : null;

  return (
    <th 
      className="p-4 font-semibold cursor-pointer hover:bg-cyan-900/20 transition-colors select-none"
      onClick={() => onSort(field)}
    >
      <div className="flex items-center">
        {children}
        {SortIcon}
      </div>
    </th>
  );
};