import type { Driver } from '../../types/Driver';
import { DriverHoverCard } from './DriverHoverCard';

interface RegressionTableRowProps {
  driver: Driver;
}

export const RegressionTableRow = ({ driver }: RegressionTableRowProps) => {
  const predictedPos = driver.predicted_position || 0;
  const currentPos = driver.position;
  const positionChange = currentPos - predictedPos;

  const getPositionChangeColor = () => {
    if (positionChange > 0) return 'text-green-400'; // Moving up
    if (positionChange < 0) return 'text-red-400'; // Moving down
    return 'text-gray-400'; // No change
  };

  const getPositionChangeSymbol = () => {
    if (positionChange > 0) return '↑';
    if (positionChange < 0) return '↓';
    return '=';
  };

  return (
    <tr className="border-b border-gray-700/30 hover:bg-gray-800/20 transition-colors duration-200">
      <td className="px-6 py-4">
        <DriverHoverCard driver={driver}>
          <div className="flex items-center gap-3">
            <span className="text-white font-medium hover:text-cyan-300 transition-colors cursor-pointer">
              {driver.driver}
            </span>
          </div>
        </DriverHoverCard>
      </td>
      <td className="px-6 py-4">
        <span className="text-white font-medium">{currentPos}</span>
      </td>
      <td className="px-6 py-4">
        <span className="text-white">{driver.points}</span>
      </td>
      <td className="px-6 py-4">
        <span className="text-white">
          {(driver.win_rate * 100).toFixed(1)}%
        </span>
      </td>
      <td className="px-6 py-4">
        <span className="text-white">
          {(driver.top_10_rate * 100).toFixed(1)}%
        </span>
      </td>
      <td className="px-6 py-4">
        <span className="text-white">
          {(driver.dnf_rate * 100).toFixed(1)}%
        </span>
      </td>
      <td className="px-6 py-4">
        <span className="text-white">
          {(driver.participation_rate * 100).toFixed(1)}%
        </span>
      </td>
      <td className="px-6 py-4">
        <span className="text-white">{driver.experience}</span>
      </td>
      <td className="px-6 py-4">
        <div className="flex items-center gap-2">
          <span className="text-2xl font-bold text-cyan-400">
            {Math.round(predictedPos)}
          </span>
          <span className={`text-sm font-medium ${getPositionChangeColor()}`}>
            {getPositionChangeSymbol()} {Math.abs(positionChange).toFixed(1)}
          </span>
        </div>
      </td>
    </tr>
  );
};
