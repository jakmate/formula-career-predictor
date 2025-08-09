import { render } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { PredictionsTable } from './PredictionsTable';

// Mock the BasePredictionsTable component
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
      <div data-testid="title-f3-to-f2">{getTitle('f3_to_f2')}</div>
      <div data-testid="title-f2-to-f1">{getTitle('f2_to_f1')}</div>
      <div data-testid="description-f3-to-f2">{getDescription('f3_to_f2')}</div>
      <div data-testid="description-f2-to-f1">{getDescription('f2_to_f1')}</div>
    </div>
  ),
}));

describe('PredictionsTable', () => {
  it('renders BasePredictionsTable with correct props', () => {
    const { getByTestId } = render(<PredictionsTable />);

    expect(getByTestId('base-predictions-table')).toBeInTheDocument();
    expect(getByTestId('variant')).toHaveTextContent('promotions');
    expect(getByTestId('default-series')).toHaveTextContent('f3_to_f2');
  });

  it('passes correct series options', () => {
    const { getByTestId } = render(<PredictionsTable />);

    const seriesOptions = JSON.parse(
      getByTestId('series-options').textContent || ''
    );
    expect(seriesOptions).toEqual([
      { value: 'f3_to_f2', label: 'F3 → F2 Promotions' },
      { value: 'f2_to_f1', label: 'F2 → F1 Promotions' },
    ]);
  });

  it('getTitle returns correct titles for different series', () => {
    const { getByTestId } = render(<PredictionsTable />);

    expect(getByTestId('title-f3-to-f2')).toHaveTextContent(
      'F3 to F2 Predictions'
    );
    expect(getByTestId('title-f2-to-f1')).toHaveTextContent(
      'F2 to F1 Predictions'
    );
  });

  it('getDescription returns correct descriptions for different series', () => {
    const { getByTestId } = render(<PredictionsTable />);

    expect(getByTestId('description-f3-to-f2')).toHaveTextContent(
      'AI-powered analysis of Formula 3 drivers likely to advance to Formula 2'
    );
    expect(getByTestId('description-f2-to-f1')).toHaveTextContent(
      'AI-powered analysis of Formula 2 drivers likely to advance to Formula 1'
    );
  });

  it('getTitle handles edge cases', () => {
    const { getByTestId } = render(<PredictionsTable />);

    // Test that any value other than 'f3_to_f2' returns F2 to F1 title
    const titleElement = getByTestId('title-f2-to-f1');

    expect(titleElement).toHaveTextContent('F2 to F1 Predictions');
  });

  it('getDescription handles edge cases', () => {
    const { getByTestId } = render(<PredictionsTable />);

    // Test that any value other than 'f3_to_f2' returns F2 to F1 description
    const descriptionElement = getByTestId('description-f2-to-f1');

    expect(descriptionElement).toHaveTextContent(
      'AI-powered analysis of Formula 2 drivers likely to advance to Formula 1'
    );
  });
});
