import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { BasePredictionsTable } from './BasePredictionsTable';

// Mock components
vi.mock('../ErrorDisplay', () => ({
  ErrorDisplay: ({ error }: { error: string }) => (
    <div data-testid="error-display">{error}</div>
  ),
}));

vi.mock('./TableContent', () => ({
  BaseTableContent: ({
    predictions,
    variant,
  }: {
    predictions: unknown[];
    variant: string;
  }) => (
    <div data-testid="table-content">
      <div data-testid="predictions-count">{predictions.length}</div>
      <div data-testid="table-variant">{variant}</div>
    </div>
  ),
}));

interface HeaderProps {
  title: string;
  description: string;
  rightContent: React.ReactNode;
  bottomContent: React.ReactNode;
}

vi.mock('../Header', () => ({
  Header: ({
    title,
    description,
    rightContent,
    bottomContent,
  }: HeaderProps) => (
    <div data-testid="header">
      <div data-testid="title">{title}</div>
      <div data-testid="description">{description}</div>
      <div data-testid="right-content">{rightContent}</div>
      <div data-testid="bottom-content">{bottomContent}</div>
    </div>
  ),
}));

// Mock the hook
const mockUsePredictions = vi.fn();
vi.mock('../../hooks/usePredictions', () => ({
  usePredictions: (selectedSeries: string) =>
    mockUsePredictions(selectedSeries),
}));

describe('BasePredictionsTable', () => {
  const defaultProps = {
    variant: 'promotions' as const,
    defaultSeries: 'f3_to_f2',
    seriesOptions: [
      { value: 'f3_to_f2', label: 'F3 → F2' },
      { value: 'f2_to_f1', label: 'F2 → F1' },
    ],
    getTitle: (series: string) => `Title for ${series}`,
    getDescription: (series: string) => `Description for ${series}`,
  };

  const mockHookReturn = {
    predictions: { model1: { predictions: [{ id: 1 }, { id: 2 }] } },
    selectedModel: 'model1',
    setSelectedModel: vi.fn(),
    models: ['model1', 'model2'],
    loading: false,
    status: {
      last_scrape: '2023-01-01T00:00:00Z',
      last_training: '2023-01-02T00:00:00Z',
      models_available: ['model1', 'model2'],
    },
    error: null,
    handleRefresh: vi.fn(),
    currentPredictions: [{ id: 1 }, { id: 2 }],
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockUsePredictions.mockReturnValue(mockHookReturn);
  });

  it('renders with correct initial state', () => {
    render(<BasePredictionsTable {...defaultProps} />);

    expect(screen.getByTestId('title')).toHaveTextContent('Title for f3_to_f2');
    expect(screen.getByTestId('description')).toHaveTextContent(
      'Description for f3_to_f2'
    );
  });

  it('maps variant correctly to table variant', () => {
    const { rerender } = render(
      <BasePredictionsTable {...defaultProps} variant="positions" />
    );
    expect(screen.getByTestId('table-variant')).toHaveTextContent('regression');

    rerender(<BasePredictionsTable {...defaultProps} variant="promotions" />);
    expect(screen.getByTestId('table-variant')).toHaveTextContent('default');
  });

  it('shows loading state when loading with no current predictions', () => {
    mockUsePredictions.mockReturnValue({
      ...mockHookReturn,
      loading: true,
      currentPredictions: [],
    });

    render(<BasePredictionsTable {...defaultProps} />);
    expect(screen.getByText('Loading predictions...')).toBeInTheDocument();
  });

  it('shows no predictions message when not loading and no predictions', () => {
    mockUsePredictions.mockReturnValue({
      ...mockHookReturn,
      loading: false,
      currentPredictions: [],
    });

    render(<BasePredictionsTable {...defaultProps} />);
    expect(
      screen.getByText(
        'No predictions available. Select a model and refresh data.'
      )
    ).toBeInTheDocument();
  });

  it('renders table content when predictions are available', () => {
    render(<BasePredictionsTable {...defaultProps} />);

    expect(screen.getByTestId('table-content')).toBeInTheDocument();
    expect(screen.getByTestId('predictions-count')).toHaveTextContent('2');
  });

  it('shows error when error exists', () => {
    mockUsePredictions.mockReturnValue({
      ...mockHookReturn,
      error: 'Test error',
    });

    render(<BasePredictionsTable {...defaultProps} />);
    expect(screen.getByTestId('error-display')).toHaveTextContent('Test error');
  });

  it('handles series selection change', () => {
    render(<BasePredictionsTable {...defaultProps} />);

    const seriesSelect = screen.getByDisplayValue('F3 → F2');
    fireEvent.change(seriesSelect, { target: { value: 'f2_to_f1' } });

    // Should trigger re-render with new title/description
    expect(screen.getByTestId('title')).toHaveTextContent('Title for f2_to_f1');
  });

  it('handles model selection change', () => {
    render(<BasePredictionsTable {...defaultProps} />);

    const modelSelect = screen.getByDisplayValue('model1');
    fireEvent.change(modelSelect, { target: { value: 'model2' } });

    expect(mockHookReturn.setSelectedModel).toHaveBeenCalledWith('model2');
  });

  it('handles refresh button click', () => {
    render(<BasePredictionsTable {...defaultProps} />);

    const refreshButton = screen.getByText('Refresh');
    fireEvent.click(refreshButton);

    expect(mockHookReturn.handleRefresh).toHaveBeenCalled();
  });

  it('shows loading state on refresh button when loading', () => {
    mockUsePredictions.mockReturnValue({
      ...mockHookReturn,
      loading: true,
    });

    render(<BasePredictionsTable {...defaultProps} />);
    expect(screen.getByText('Updating...')).toBeInTheDocument();
  });
});
