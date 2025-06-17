interface ModelInfoProps {
  selectedModel: string;
  predictionCount: number;
}

export const ModelInfo = ({ selectedModel, predictionCount }: ModelInfoProps) => (
  <div className="mt-6 bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/20">
    <h3 className="text-white font-semibold mb-2">Model: {selectedModel}</h3>
    <p className="text-blue-200 text-sm">
      Total predictions: {predictionCount} drivers
    </p>
  </div>
);