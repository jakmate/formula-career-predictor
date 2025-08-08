import type { Driver } from '../../types/Driver';
import { TableHeader } from './TableHeader';
import { useSorting } from '../../hooks/useSorting';
import { BaseTableRow } from './TableRow';

interface BaseTableContentProps {
  predictions: Driver[];
  variant?: 'default' | 'regression';
}

export const BaseTableContent = ({
  predictions,
  variant = 'default',
}: BaseTableContentProps) => {
  const { sortedData, sortConfig, handleSort } = useSorting(predictions);

  const getLastColumnHeader = () => {
    if (variant === 'regression') return 'Predicted Position';
    return 'F2 Probability';
  };

  const getLastColumnField = () => {
    if (variant === 'regression') return 'predicted_position';
    return 'empirical_percentage';
  };

  const getPositionHeader = () => {
    if (variant === 'regression') return 'Current Position';
    return 'Position';
  };

  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead className="bg-gray-900/50">
          <tr className="text-left text-white/90">
            <TableHeader
              field="driver"
              sortConfig={sortConfig}
              onSort={handleSort}
            >
              Driver
            </TableHeader>
            <TableHeader
              field="position"
              sortConfig={sortConfig}
              onSort={handleSort}
            >
              {getPositionHeader()}
            </TableHeader>
            <TableHeader
              field="points"
              sortConfig={sortConfig}
              onSort={handleSort}
            >
              Points
            </TableHeader>
            <TableHeader
              field="win_rate"
              sortConfig={sortConfig}
              onSort={handleSort}
            >
              Win %
            </TableHeader>
            <TableHeader
              field="top_10_rate"
              sortConfig={sortConfig}
              onSort={handleSort}
            >
              Top 10 %
            </TableHeader>
            <TableHeader
              field="dnf_rate"
              sortConfig={sortConfig}
              onSort={handleSort}
            >
              DNF %
            </TableHeader>
            <TableHeader
              field="participation_rate"
              sortConfig={sortConfig}
              onSort={handleSort}
            >
              Participation %
            </TableHeader>
            <TableHeader
              field="experience"
              sortConfig={sortConfig}
              onSort={handleSort}
            >
              Experience
            </TableHeader>
            <TableHeader
              field={getLastColumnField()}
              sortConfig={sortConfig}
              onSort={handleSort}
            >
              {getLastColumnHeader()}
            </TableHeader>
          </tr>
        </thead>
        <tbody>
          {sortedData.map((driver: Driver) => (
            <BaseTableRow
              key={driver.driver}
              driver={driver}
              variant={variant}
            />
          ))}
        </tbody>
      </table>
    </div>
  );
};
