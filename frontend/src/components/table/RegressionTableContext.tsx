import type { Driver } from '../../types/Driver';
import { TableHeader } from './TableHeader';
import { useSorting } from '../../hooks/useSorting';
import { RegressionTableRow } from './RegressionTableRow';

interface RegressionTableContentProps {
  predictions: Driver[];
}

export const RegressionTableContent = ({
  predictions,
}: RegressionTableContentProps) => {
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
              Current Position
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
              field="predicted_position"
              sortConfig={sortConfig}
              onSort={handleSort}
            >
              Predicted Position
            </TableHeader>
          </tr>
        </thead>
        <tbody>
          {sortedData.map((driver: Driver) => (
            <RegressionTableRow key={driver.driver} driver={driver} />
          ))}
        </tbody>
      </table>
    </div>
  );
};
