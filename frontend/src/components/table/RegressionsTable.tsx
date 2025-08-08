import { BasePredictionsTable } from './BasePredictionsTable';

export const RegressionTable = () => {
  const seriesOptions = [
    { value: 'f3_regression', label: 'F3 Position Predictions' },
    { value: 'f2_regression', label: 'F2 Position Predictions' },
    { value: 'f1_regression', label: 'F1 Position Predictions' },
  ];

  const getTitle = (selectedSeries: string) => {
    switch (selectedSeries) {
      case 'f3_regression':
        return 'F3 Position Predictions';
      case 'f2_regression':
        return 'F2 Position Predictions';
      case 'f1_regression':
        return 'F1 Position Predictions';
      default:
        return 'Position Predictions';
    }
  };

  const getDescription = (selectedSeries: string) => {
    switch (selectedSeries) {
      case 'f3_regression':
        return 'AI-powered predictions of final championship positions for Formula 3 drivers';
      case 'f2_regression':
        return 'AI-powered predictions of final championship positions for Formula 2 drivers';
      case 'f1_regression':
        return 'AI-powered predictions of final championship positions for Formula 1 drivers';
      default:
        return 'AI-powered position predictions';
    }
  };

  return (
    <BasePredictionsTable
      variant="positions"
      defaultSeries="f3_regression"
      seriesOptions={seriesOptions}
      getTitle={getTitle}
      getDescription={getDescription}
    />
  );
};
