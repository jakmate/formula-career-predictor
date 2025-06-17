import type { Driver } from '../../types/Driver';
import { TableHeader } from './TableHeader';
import { TableRow } from './TableRow';
import { useSorting } from '../../hooks/useSorting';

interface TableContentProps {
  predictions: Driver[];
}

export const TableContent = ({ predictions }: TableContentProps) => {
  const { sortedData, sortConfig, handleSort } = useSorting(predictions);

  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead className="bg-white/5">
          <tr className="text-left text-white/90">
            <TableHeader field="driver" sortConfig={sortConfig} onSort={handleSort}>
              Driver
            </TableHeader>
            <th className="p-4 font-semibold">Nat.</th>
            <TableHeader field="position" sortConfig={sortConfig} onSort={handleSort}>
              Position
            </TableHeader>
            <TableHeader field="points" sortConfig={sortConfig} onSort={handleSort}>
              Points
            </TableHeader>
            <TableHeader field="win_rate" sortConfig={sortConfig} onSort={handleSort}>
              Win %
            </TableHeader>
            <TableHeader field="podium_rate" sortConfig={sortConfig} onSort={handleSort}>
              Podium %
            </TableHeader>
            <TableHeader field="top_10_rate" sortConfig={sortConfig} onSort={handleSort}>
              Top 10 %
            </TableHeader>
            <TableHeader field="experience" sortConfig={sortConfig} onSort={handleSort}>
              Experience
            </TableHeader>
            <TableHeader field="empirical_percentage" sortConfig={sortConfig} onSort={handleSort}>
              F2 Probability
            </TableHeader>
            <TableHeader field="prediction" sortConfig={sortConfig} onSort={handleSort}>
              Prediction
            </TableHeader>
          </tr>
        </thead>
        <tbody>
          {sortedData.map((driver: Driver) => (
            <TableRow key={driver.driver} driver={driver} />
          ))}
        </tbody>
      </table>
    </div>
  );
};