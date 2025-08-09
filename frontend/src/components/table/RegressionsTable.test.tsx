import { render } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { RegressionTable } from './RegressionsTable';

interface MockProps {
  variant: string;
  defaultSeries: string;
  seriesOptions: Array<{ value: string; label: string }>;
  getTitle: (selectedSeries: string) => string;
  getDescription: (selectedSeries: string) => string;
}

vi.mock('./BasePredictionsTable', () => ({
  BasePredictionsTable: ({
    variant,
    defaultSeries,
    seriesOptions,
    getTitle,
    getDescription,
  }: MockProps) => (
    <div data-testid="base-predictions-table">
      <div data-testid="variant">{variant}</div>
      <div data-testid="default-series">{defaultSeries}</div>
      <div data-testid="series-options">{JSON.stringify(seriesOptions)}</div>
      <div data-testid="title-f3">{getTitle('f3_regression')}</div>
      <div data-testid="title-f2">{getTitle('f2_regression')}</div>
      <div data-testid="title-f1">{getTitle('f1_regression')}</div>
      <div data-testid="title-default">{getTitle('unknown')}</div>
      <div data-testid="description-f3">{getDescription('f3_regression')}</div>
      <div data-testid="description-f2">{getDescription('f2_regression')}</div>
      <div data-testid="description-f1">{getDescription('f1_regression')}</div>
      <div data-testid="description-default">{getDescription('unknown')}</div>
    </div>
  ),
}));

describe('RegressionTable', () => {
  it('renders with correct props', () => {
    const { getByTestId } = render(<RegressionTable />);

    expect(getByTestId('variant')).toHaveTextContent('positions');
    expect(getByTestId('default-series')).toHaveTextContent('f3_regression');
  });

  it('passes correct series options', () => {
    const { getByTestId } = render(<RegressionTable />);

    const seriesOptions = JSON.parse(
      getByTestId('series-options').textContent || ''
    );
    expect(seriesOptions).toEqual([
      { value: 'f3_regression', label: 'F3 Position Predictions' },
      { value: 'f2_regression', label: 'F2 Position Predictions' },
      { value: 'f1_regression', label: 'F1 Position Predictions' },
    ]);
  });

  describe('getTitle', () => {
    it('returns correct titles for all series', () => {
      const { getByTestId } = render(<RegressionTable />);

      expect(getByTestId('title-f3')).toHaveTextContent(
        'F3 Position Predictions'
      );
      expect(getByTestId('title-f2')).toHaveTextContent(
        'F2 Position Predictions'
      );
      expect(getByTestId('title-f1')).toHaveTextContent(
        'F1 Position Predictions'
      );
    });

    it('returns default title for unknown series', () => {
      const { getByTestId } = render(<RegressionTable />);

      expect(getByTestId('title-default')).toHaveTextContent(
        'Position Predictions'
      );
    });
  });

  describe('getDescription', () => {
    it('returns correct descriptions for all series', () => {
      const { getByTestId } = render(<RegressionTable />);

      expect(getByTestId('description-f3')).toHaveTextContent(
        'AI-powered predictions of final championship positions for Formula 3 drivers'
      );
      expect(getByTestId('description-f2')).toHaveTextContent(
        'AI-powered predictions of final championship positions for Formula 2 drivers'
      );
      expect(getByTestId('description-f1')).toHaveTextContent(
        'AI-powered predictions of final championship positions for Formula 1 drivers'
      );
    });

    it('returns default description for unknown series', () => {
      const { getByTestId } = render(<RegressionTable />);

      expect(getByTestId('description-default')).toHaveTextContent(
        'AI-powered position predictions'
      );
    });
  });
});
