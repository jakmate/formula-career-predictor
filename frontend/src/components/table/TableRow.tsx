import type { Driver } from "../../types/Driver";
import { ProbabilityBar } from "../ProbabilityBar";
import { DriverHoverCard } from "./DriverHoverCard";

interface TableRowProps {
  driver: Driver;
}

export const TableRow = ({ driver }: TableRowProps) => (
  <tr
    className={`border-t border-cyan-500/10 hover:bg-cyan-900/10 transition-colors ${
      driver.prediction === 1
        ? "bg-gradient-to-r from-green-900/20 to-cyan-900/20"
        : ""
    }`}
  >
    <td className="p-4 text-white font-medium">
      <DriverHoverCard driver={driver}>
        <span className="cursor-pointer hover:text-cyan-300 transition-colors">
          {driver.driver}
        </span>
      </DriverHoverCard>
    </td>
    <td className="p-4 text-white">#{driver.position}</td>
    <td className="p-4 text-white">{driver.points.toFixed(1)}</td>
    <td className="p-4 text-white">{(driver.win_rate * 100).toFixed(1)}%</td>
    <td className="p-4 text-white">{(driver.podium_rate * 100).toFixed(1)}%</td>
    <td className="p-4 text-white">{(driver.top_10_rate * 100).toFixed(1)}%</td>
    <td className="p-4 text-white">{(driver.dnf_rate * 100).toFixed(1)}%</td>
    <td className="p-4 text-white">
      {(driver.participation_rate * 100).toFixed(1)}%
    </td>
    <td className="p-4 text-white">{driver.experience} years</td>
    <td className="p-4">
      <ProbabilityBar percentage={driver.empirical_percentage} />
    </td>
  </tr>
);
