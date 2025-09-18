import type { Driver } from '../../types/Driver';
import { ProbabilityBar } from '../ProbabilityBar';
import { DriverHoverCard } from './DriverHoverCard';

interface BaseTableRowProps {
  driver: Driver;
  className?: string;
}

export const BaseTableRow = ({ driver, className = '' }: BaseTableRowProps) => {
  const empiricalPercentage = driver.empirical_percentage ?? 0;
  const formatPercentage = (value: number) => (value * 100).toFixed(1) + '%';

  const baseClasses = `border-t border-cyan-500/10 hover:bg-cyan-900/10 transition-colors ${
    empiricalPercentage >= 50
      ? 'bg-gradient-to-r from-green-900/20 to-cyan-900/20'
      : ''
  }`;

  return (
    <tr className={`${baseClasses} ${className}`}>
      <td className="p-4">
        <DriverHoverCard driver={driver}>
          <div className="flex items-center gap-3">
            <span className="text-white font-medium hover:text-cyan-300 transition-colors cursor-pointer">
              {driver.driver}
            </span>
          </div>
        </DriverHoverCard>
      </td>
      <td className="p-4">
        <span className="text-white font-medium">#{driver.position}</span>
      </td>
      <td className="p-4">
        <span className="text-white">{driver.points.toFixed(1)}</span>
      </td>
      <td className="p-4">
        <span className="text-white">{formatPercentage(driver.win_rate)}</span>
      </td>
      <td className="p-4">
        <span className="text-white">
          {formatPercentage(driver.top_10_rate)}
        </span>
      </td>
      <td className="p-4">
        <span className="text-white">{formatPercentage(driver.dnf_rate)}</span>
      </td>
      <td className="p-4">
        <span className="text-white">
          {formatPercentage(driver.participation_rate)}
        </span>
      </td>
      <td className="p-4">
        <span className="text-white">
          {driver.experience}
          {' years'}
        </span>
      </td>
      <td className="p-4">
        <ProbabilityBar percentage={empiricalPercentage} />
      </td>
    </tr>
  );
};
