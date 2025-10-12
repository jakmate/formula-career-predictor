import type { Driver } from '../../types/Driver';
import { TableHeader } from './TableHeader';
import { useSorting } from '../../hooks/useSorting';
import { BaseTableRow } from './TableRow';

interface BaseTableContentProps {
  predictions: Driver[];
}

export const BaseTableContent = ({ predictions }: BaseTableContentProps) => {
  const { sortedData, sortConfig, handleSort } = useSorting(predictions);

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
              Position
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
              field="empirical_percentage"
              sortConfig={sortConfig}
              onSort={handleSort}
            >
              F2 Probability
            </TableHeader>
          </tr>
        </thead>
        <tbody>
          {sortedData.map((driver: Driver) => (
            <BaseTableRow key={driver.driver} driver={driver} />
          ))}
        </tbody>
      </table>
    </div>
  );
};
