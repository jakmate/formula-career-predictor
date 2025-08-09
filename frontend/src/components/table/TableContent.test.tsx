import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import type { Driver } from '../../types/Driver';
import { BaseTableContent } from './TableContent';

// Mock dependencies
vi.mock('../TableHeader', () => ({
  TableHeader: ({
    children,
    field,
    onSort,
  }: {
    children: React.ReactNode;
    field: string;
    sortConfig: { field: string | null; direction: 'asc' | 'desc' };
    onSort: (field: string) => void;
  }) => (
    <th data-testid={`header-${field}`} onClick={() => onSort(field)}>
      {children}
    </th>
  ),
}));

// Don't mock BaseTableRow - test with real component

vi.mock('../../../hooks/useSorting', () => ({
  useSorting: vi.fn((data) => ({
    sortedData: data,
    sortConfig: { field: null, direction: 'asc' },
    handleSort: vi.fn(),
  })),
}));

const mockDrivers: Driver[] = [
  {
    driver: 'Max Verstappen',
    position: 1,
    points: 575,
    win_rate: 0.65,
    top_10_rate: 0.95,
    dnf_rate: 0.05,
    participation_rate: 1.0,
    experience: 9,
    empirical_percentage: 0.85,
    predicted_position: 1.2,
    wins: 0,
    podiums: 0,
    nationality: '',
    teammate_h2h: 0,
    team: '',
    team_pos: 0,
    team_points: 0,
  },
  {
    driver: 'Lewis Hamilton',
    position: 2,
    points: 450,
    win_rate: 0.45,
    top_10_rate: 0.88,
    dnf_rate: 0.08,
    participation_rate: 0.98,
    experience: 17,
    empirical_percentage: 0.72,
    predicted_position: 2.1,
    wins: 0,
    podiums: 0,
    nationality: '',
    teammate_h2h: 0,
    team: '',
    team_pos: 0,
    team_points: 0,
  },
];

describe('BaseTableContent', () => {
  it('renders with default variant', () => {
    render(<BaseTableContent predictions={mockDrivers} />);

    expect(screen.getByText('Driver')).toBeInTheDocument();
    expect(screen.getByText('Position')).toBeInTheDocument();
    expect(screen.getByText('F2 Probability')).toBeInTheDocument();
    expect(screen.queryByText('Current Position')).not.toBeInTheDocument();
    expect(screen.queryByText('Predicted Position')).not.toBeInTheDocument();
  });

  it('renders with regression variant', () => {
    render(<BaseTableContent predictions={mockDrivers} variant="regression" />);

    expect(screen.getByText('Current Position')).toBeInTheDocument();
    expect(screen.getByText('Predicted Position')).toBeInTheDocument();
    expect(screen.queryByText('Position')).not.toBeInTheDocument();
    expect(screen.queryByText('F2 Probability')).not.toBeInTheDocument();
  });

  it('renders all standard headers', () => {
    render(<BaseTableContent predictions={mockDrivers} />);

    expect(screen.getByText('Driver')).toBeInTheDocument();
    expect(screen.getByText('Points')).toBeInTheDocument();
    expect(screen.getByText('Win %')).toBeInTheDocument();
    expect(screen.getByText('Top 10 %')).toBeInTheDocument();
    expect(screen.getByText('DNF %')).toBeInTheDocument();
    expect(screen.getByText('Participation %')).toBeInTheDocument();
    expect(screen.getByText('Experience')).toBeInTheDocument();
  });

  it('renders correct number of table headers', () => {
    render(<BaseTableContent predictions={mockDrivers} />);

    const headers = screen.getAllByRole('columnheader');
    expect(headers).toHaveLength(9); // 9 columns total
  });

  it('renders table structure correctly', () => {
    render(<BaseTableContent predictions={mockDrivers} />);

    const table = screen.getByRole('table');
    expect(table).toBeInTheDocument();
    expect(table).toHaveClass('w-full');

    const thead = table.querySelector('thead');
    expect(thead).toBeInTheDocument();
    expect(thead).toHaveClass('bg-gray-900/50');

    const tbody = table.querySelector('tbody');
    expect(tbody).toBeInTheDocument();
  });

  it('renders driver rows with correct variant', () => {
    render(<BaseTableContent predictions={mockDrivers} variant="regression" />);

    // Check that drivers are rendered by looking for their names
    expect(screen.getByText('Max Verstappen')).toBeInTheDocument();
    expect(screen.getByText('Lewis Hamilton')).toBeInTheDocument();
  });

  it('renders driver rows with default variant when not specified', () => {
    render(<BaseTableContent predictions={mockDrivers} />);

    // Check that drivers are rendered
    expect(screen.getByText('Max Verstappen')).toBeInTheDocument();
    expect(screen.getByText('Lewis Hamilton')).toBeInTheDocument();
  });

  it('handles empty predictions array', () => {
    render(<BaseTableContent predictions={[]} />);

    const table = screen.getByRole('table');
    expect(table).toBeInTheDocument();

    // Headers should still be present
    expect(screen.getByText('Driver')).toBeInTheDocument();

    // No driver names should be rendered
    expect(screen.queryByText('Max Verstappen')).not.toBeInTheDocument();
  });

  it('renders overflow container', () => {
    const { container } = render(
      <BaseTableContent predictions={mockDrivers} />
    );

    const overflowDiv = container.querySelector('.overflow-x-auto');
    expect(overflowDiv).toBeInTheDocument();
  });

  it('renders all header columns in correct order', () => {
    render(<BaseTableContent predictions={mockDrivers} />);

    const headers = screen.getAllByRole('columnheader');
    const expectedHeaders = [
      'Driver',
      'Position',
      'Points',
      'Win %',
      'Top 10 %',
      'DNF %',
      'Participation %',
      'Experience',
      'F2 Probability',
    ];

    expect(headers).toHaveLength(expectedHeaders.length);
    expectedHeaders.forEach((headerText, index) => {
      expect(headers[index]).toHaveTextContent(headerText);
    });
  });

  it('renders all header columns for regression variant in correct order', () => {
    render(<BaseTableContent predictions={mockDrivers} variant="regression" />);

    const headers = screen.getAllByRole('columnheader');
    const expectedHeaders = [
      'Driver',
      'Current Position',
      'Points',
      'Win %',
      'Top 10 %',
      'DNF %',
      'Participation %',
      'Experience',
      'Predicted Position',
    ];

    expect(headers).toHaveLength(expectedHeaders.length);
    expectedHeaders.forEach((headerText, index) => {
      expect(headers[index]).toHaveTextContent(headerText);
    });
  });

  it('applies correct CSS classes to table row', () => {
    const { container } = render(
      <BaseTableContent predictions={mockDrivers} />
    );

    const headerRow = container.querySelector('thead tr');
    expect(headerRow).toHaveClass('text-left', 'text-white/90');
  });
});
