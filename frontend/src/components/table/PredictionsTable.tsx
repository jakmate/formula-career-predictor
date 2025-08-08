import { BasePredictionsTable } from './BasePredictionsTable';

export const PredictionsTable = () => {
  const seriesOptions = [
    { value: 'f3_to_f2', label: 'F3 → F2 Promotions' },
    { value: 'f2_to_f1', label: 'F2 → F1 Promotions' },
  ];

  const getTitle = (selectedSeries: string) => {
    return selectedSeries === 'f3_to_f2'
      ? 'F3 to F2 Predictions'
      : 'F2 to F1 Predictions';
  };

  const getDescription = (selectedSeries: string) => {
    return `AI-powered analysis of ${
      selectedSeries === 'f3_to_f2'
        ? 'Formula 3 drivers likely to advance to Formula 2'
        : 'Formula 2 drivers likely to advance to Formula 1'
    }`;
  };

  return (
    <BasePredictionsTable
      variant="promotions"
      defaultSeries="f3_to_f2"
      seriesOptions={seriesOptions}
      getTitle={getTitle}
      getDescription={getDescription}
    />
  );
};
