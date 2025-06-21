import type { Driver } from '../../types/Driver';
import { ProbabilityBar } from '../ProbabilityBar';
import { DriverHoverCard } from './DriverHoverCard';

interface TableRowProps {
  driver: Driver;
}

export const TableRow = ({ driver }: TableRowProps) => (
  <tr 
    className={`border-t border-white/10 hover:bg-white/5 transition-colors ${
      driver.prediction === 1 ? 'bg-green-500/10' : ''
    }`}
  >
    <td className="p-4 text-white font-medium">
      <DriverHoverCard driver={driver}>
        <span className="cursor-pointer hover:text-blue-300 transition-colors">
          {driver.driver}
        </span>
      </DriverHoverCard>
    </td>
    <td className="p-4 text-white">{driver.position}</td>
    <td className="p-4 text-white">{driver.points.toFixed(1)}</td>
    <td className="p-4 text-white">{(driver.win_rate * 100).toFixed(1)}%</td>
    <td className="p-4 text-white">{(driver.podium_rate * 100).toFixed(1)}%</td>
    <td className="p-4 text-white">{(driver.top_10_rate * 100).toFixed(1)}%</td>
    <td className="p-4 text-white">{(driver.dnf_rate * 100).toFixed(1)}%</td>
    <td className="p-4 text-white">{(driver.participation_rate * 100).toFixed(1)}%</td>
    <td className="p-4 text-white">{driver.experience} years</td>
    <td className="p-4">
      <ProbabilityBar percentage={driver.empirical_percentage} />
    </td>
  </tr>
);