interface ErrorDisplayProps {
  error: string;
}

export const ErrorDisplay = ({ error }: ErrorDisplayProps) => (
  <div className="bg-red-500/20 border border-red-500/50 rounded-lg p-4 mb-6 text-red-200 backdrop-blur-sm">
    {error}
  </div>
);