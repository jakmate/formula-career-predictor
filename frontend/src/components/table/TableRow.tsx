import type { Driver } from '../../types/Driver';
import { getFlagComponent } from '../../utils/flags';
import { ProbabilityBar } from '../ProbabilityBar';

interface TableRowProps {
  driver: Driver;
}

export const TableRow = ({ driver }: TableRowProps) => (
  <tr 
    className={`border-t border-white/10 hover:bg-white/5 transition-colors ${
      driver.prediction === 1 ? 'bg-green-500/10' : ''
    }`}
  >
    <td className="p-4 text-white font-medium">{driver.driver}</td>
    <td className="p-4">{getFlagComponent(driver.nationality)}</td>
    <td className="p-4 text-white">{driver.position}</td>
    <td className="p-4 text-white">{driver.points.toFixed(1)}</td>
    <td className="p-4 text-white">{(driver.win_rate * 100).toFixed(1)}%</td>
    <td className="p-4 text-white">{(driver.podium_rate * 100).toFixed(1)}%</td>
    <td className="p-4 text-white">{(driver.top_10_rate * 100).toFixed(1)}%</td>
    <td className="p-4 text-white">{driver.experience} years</td>
    <td className="p-4">
      <ProbabilityBar percentage={driver.empirical_percentage} />
    </td>
    <td className="p-4">
      <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
        driver.prediction === 1 
          ? 'bg-green-500/20 text-green-300 border border-green-500/30' 
          : 'bg-red-500/20 text-red-300 border border-red-500/30'
      }`}>
        {driver.prediction === 1 ? 'Likely F2' : 'Unlikely F2'}
      </span>
    </td>
  </tr>
);