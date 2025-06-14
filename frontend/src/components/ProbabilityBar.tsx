interface ProbabilityBarProps {
  percentage: number;
}

export const ProbabilityBar = ({ percentage }: ProbabilityBarProps)=>(
    <div className="flex items-center gap-2">
        <div className="w-20 h-2 bg-white/20 rounded-full overflow-hidden">
        <div 
            className={`h-full transition-all duration-500 ${
            percentage > 70 ? 'bg-green-400' :
            percentage > 40 ? 'bg-yellow-400' : 'bg-red-400'
            }`}
            style={{ width: `${Math.min(percentage, 100)}%` }}
        />
        </div>
        <span className="text-white text-sm font-medium">
        {percentage.toFixed(1)}%
        </span>
    </div>
);