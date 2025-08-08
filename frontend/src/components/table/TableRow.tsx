import type { Driver } from '../../types/Driver';
import { ProbabilityBar } from '../ProbabilityBar';
import { DriverHoverCard } from './DriverHoverCard';

interface BaseTableRowProps {
  driver: Driver;
  variant?: 'default' | 'regression';
  className?: string;
}

export const BaseTableRow = ({
  driver,
  variant = 'default',
  className = '',
}: BaseTableRowProps) => {
  const empiricalPercentage = driver.empirical_percentage ?? 0;
  const predictedPos = driver.predicted_position || 0;
  const currentPos = driver.position;
  const positionChange = currentPos - predictedPos;

  const getPositionChangeColor = () => {
    if (positionChange > 0) return 'text-green-400';
    if (positionChange < 0) return 'text-red-400';
    return 'text-gray-400';
  };

  const getPositionChangeSymbol = () => {
    if (positionChange > 0) return '↑';
    if (positionChange < 0) return '↓';
    return '=';
  };

  const formatPercentage = (value: number) => (value * 100).toFixed(1) + '%';

  const baseClasses =
    variant === 'regression'
      ? 'border-b border-gray-700/30 hover:bg-gray-800/20 transition-colors duration-200'
      : `border-t border-cyan-500/10 hover:bg-cyan-900/10 transition-colors ${
          empiricalPercentage >= 50
            ? 'bg-gradient-to-r from-green-900/20 to-cyan-900/20'
            : ''
        }`;

  const cellPadding = variant === 'regression' ? 'px-6 py-4' : 'p-4';

  return (
    <tr className={`${baseClasses} ${className}`}>
      <td className={cellPadding}>
        <DriverHoverCard driver={driver}>
          <div className="flex items-center gap-3">
            <span className="text-white font-medium hover:text-cyan-300 transition-colors cursor-pointer">
              {driver.driver}
            </span>
          </div>
        </DriverHoverCard>
      </td>
      <td className={cellPadding}>
        <span className="text-white font-medium">#{driver.position}</span>
      </td>
      <td className={cellPadding}>
        <span className="text-white">
          {variant === 'regression' ? driver.points : driver.points.toFixed(1)}
        </span>
      </td>
      <td className={cellPadding}>
        <span className="text-white">{formatPercentage(driver.win_rate)}</span>
      </td>
      <td className={cellPadding}>
        <span className="text-white">
          {formatPercentage(driver.top_10_rate)}
        </span>
      </td>
      <td className={cellPadding}>
        <span className="text-white">{formatPercentage(driver.dnf_rate)}</span>
      </td>
      <td className={cellPadding}>
        <span className="text-white">
          {formatPercentage(driver.participation_rate)}
        </span>
      </td>
      <td className={cellPadding}>
        <span className="text-white">
          {driver.experience}
          {variant === 'default' ? ' years' : ''}
        </span>
      </td>
      <td className={cellPadding}>
        {variant === 'regression' ? (
          <div className="flex items-center gap-2">
            <span className="text-2xl font-bold text-cyan-400">
              {Math.round(predictedPos)}
            </span>
            <span className={`text-sm font-medium ${getPositionChangeColor()}`}>
              {getPositionChangeSymbol()} {Math.abs(positionChange).toFixed(1)}
            </span>
          </div>
        ) : (
          <ProbabilityBar percentage={empiricalPercentage} />
        )}
      </td>
    </tr>
  );
};
