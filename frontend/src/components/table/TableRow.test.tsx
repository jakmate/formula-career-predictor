import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import type { Driver } from '../../types/Driver';
import { BaseTableRow } from './TableRow';

// Mock child components
vi.mock('../ProbabilityBar', () => ({
  ProbabilityBar: ({ percentage }: { percentage: number }) => (
    <div data-testid="probability-bar">{percentage}%</div>
  ),
}));

vi.mock('./DriverHoverCard', () => ({
  DriverHoverCard: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="driver-hover-card">{children}</div>
  ),
}));

const mockDriver: Driver = {
  driver: 'John Doe',
  position: 5,
  points: 85.5,
  win_rate: 0.25,
  top_10_rate: 0.75,
  dnf_rate: 0.15,
  participation_rate: 0.9,
  experience: 2,
  empirical_percentage: 65,
  predicted_position: 3,
  wins: 0,
  podiums: 0,
  nationality: '',
  teammate_h2h: 0,
  team: '',
  team_pos: 0,
  team_points: 0,
};

describe('BaseTableRow', () => {
  it('renders driver data correctly in default variant', () => {
    render(
      <table>
        <tbody>
          <BaseTableRow driver={mockDriver} variant="default" />
        </tbody>
      </table>
    );

    expect(screen.getByText('John Doe')).toBeInTheDocument();
    expect(screen.getByText('#5')).toBeInTheDocument();
    expect(screen.getByText('85.5')).toBeInTheDocument();
    expect(screen.getByText('25.0%')).toBeInTheDocument();
    expect(screen.getByText('75.0%')).toBeInTheDocument();
    expect(screen.getByText('15.0%')).toBeInTheDocument();
    expect(screen.getByText('90.0%')).toBeInTheDocument();
    expect(screen.getByText('2 years')).toBeInTheDocument();
    expect(screen.getByTestId('probability-bar')).toBeInTheDocument();
  });

  it('renders driver data correctly in regression variant', () => {
    render(
      <table>
        <tbody>
          <BaseTableRow driver={mockDriver} variant="regression" />
        </tbody>
      </table>
    );

    expect(screen.getByText('John Doe')).toBeInTheDocument();
    expect(screen.getByText('#5')).toBeInTheDocument();
    expect(screen.getByText('85.5')).toBeInTheDocument(); // No .toFixed() in regression
    expect(screen.getByText('2')).toBeInTheDocument(); // No "years" suffix
    expect(screen.getByText('3')).toBeInTheDocument(); // Predicted position
    expect(screen.queryByTestId('probability-bar')).not.toBeInTheDocument();
  });

  it('applies correct CSS classes for default variant', () => {
    render(
      <table>
        <tbody>
          <BaseTableRow driver={mockDriver} variant="default" />
        </tbody>
      </table>
    );

    const row = screen.getByRole('row');
    expect(row).toHaveClass(
      'border-t',
      'border-cyan-500/10',
      'hover:bg-cyan-900/10'
    );
  });

  it('applies correct CSS classes for regression variant', () => {
    render(
      <table>
        <tbody>
          <BaseTableRow driver={mockDriver} variant="regression" />
        </tbody>
      </table>
    );

    const row = screen.getByRole('row');
    expect(row).toHaveClass(
      'border-b',
      'border-gray-700/30',
      'hover:bg-gray-800/20'
    );
  });

  it('applies gradient background when empirical_percentage >= 50', () => {
    const highPercentageDriver = { ...mockDriver, empirical_percentage: 75 };

    render(
      <table>
        <tbody>
          <BaseTableRow driver={highPercentageDriver} variant="default" />
        </tbody>
      </table>
    );

    const row = screen.getByRole('row');
    expect(row).toHaveClass(
      'bg-gradient-to-r',
      'from-green-900/20',
      'to-cyan-900/20'
    );
  });

  it('does not apply gradient background when empirical_percentage < 50', () => {
    const lowPercentageDriver = { ...mockDriver, empirical_percentage: 25 };

    render(
      <table>
        <tbody>
          <BaseTableRow driver={lowPercentageDriver} variant="default" />
        </tbody>
      </table>
    );

    const row = screen.getByRole('row');
    expect(row).not.toHaveClass('bg-gradient-to-r');
  });

  it('handles null empirical_percentage', () => {
    const nullPercentageDriver = { ...mockDriver, empirical_percentage: null };

    render(
      <table>
        <tbody>
          <BaseTableRow driver={nullPercentageDriver} variant="default" />
        </tbody>
      </table>
    );

    expect(screen.getByTestId('probability-bar')).toHaveTextContent('0%');
  });

  it('handles zero predicted_position', () => {
    const zeroPredictionDriver = { ...mockDriver, predicted_position: 0 };

    render(
      <table>
        <tbody>
          <BaseTableRow driver={zeroPredictionDriver} variant="regression" />
        </tbody>
      </table>
    );

    expect(screen.getByText('0')).toBeInTheDocument();
  });

  it('shows correct position change color for improvement', () => {
    const improvedDriver = {
      ...mockDriver,
      position: 2,
      predicted_position: 5,
    }; // 2 - 5 = -3 (improved)

    render(
      <table>
        <tbody>
          <BaseTableRow driver={improvedDriver} variant="regression" />
        </tbody>
      </table>
    );

    const changeSpan = screen.getByText(/↓ 3\.0/);
    expect(changeSpan).toHaveClass('text-red-400');
  });

  it('shows correct position change color for decline', () => {
    const declinedDriver = {
      ...mockDriver,
      position: 8,
      predicted_position: 3,
    }; // 8 - 3 = 5 (declined)

    render(
      <table>
        <tbody>
          <BaseTableRow driver={declinedDriver} variant="regression" />
        </tbody>
      </table>
    );

    const changeSpan = screen.getByText(/↑ 5\.0/);
    expect(changeSpan).toHaveClass('text-green-400');
  });

  it('shows correct position change color for no change', () => {
    const sameDriver = { ...mockDriver, position: 3, predicted_position: 3 };

    render(
      <table>
        <tbody>
          <BaseTableRow driver={sameDriver} variant="regression" />
        </tbody>
      </table>
    );

    const changeSpan = screen.getByText(/= 0\.0/);
    expect(changeSpan).toHaveClass('text-gray-400');
  });

  it('applies custom className when provided', () => {
    render(
      <table>
        <tbody>
          <BaseTableRow driver={mockDriver} className="custom-class" />
        </tbody>
      </table>
    );

    const row = screen.getByRole('row');
    expect(row).toHaveClass('custom-class');
  });

  it('uses correct cell padding for variants', () => {
    const { rerender } = render(
      <table>
        <tbody>
          <BaseTableRow driver={mockDriver} variant="default" />
        </tbody>
      </table>
    );

    let cells = screen.getAllByRole('cell');
    expect(cells[0]).toHaveClass('p-4');

    rerender(
      <table>
        <tbody>
          <BaseTableRow driver={mockDriver} variant="regression" />
        </tbody>
      </table>
    );

    cells = screen.getAllByRole('cell');
    expect(cells[0]).toHaveClass('px-6', 'py-4');
  });
});
