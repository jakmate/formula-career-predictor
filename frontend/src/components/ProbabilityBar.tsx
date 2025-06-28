interface ProbabilityBarProps {
  percentage: number;
}

export const ProbabilityBar = ({ percentage }: ProbabilityBarProps) => (
  <div className="flex items-center gap-2">
    <div className="w-20 h-2 bg-gray-700 rounded-full overflow-hidden">
      <div
        className={`h-full transition-all duration-500 ${
          percentage > 70
            ? "bg-gradient-to-r from-green-400 to-cyan-400"
            : percentage > 40
              ? "bg-gradient-to-r from-yellow-400 to-amber-400"
              : "bg-gradient-to-r from-red-400 to-orange-400"
        } rounded-full`}
        style={{ width: `${Math.min(percentage, 100)}%` }}
      />
    </div>
    <span className="text-white text-sm font-medium">
      {percentage.toFixed(1)}%
    </span>
  </div>
);
