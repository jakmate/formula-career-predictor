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
  dnf_rate: 0.15,
  participation_rate: 0.9,
  experience: 2,
  empirical_percentage: 65,
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
          <BaseTableRow driver={mockDriver} />
        </tbody>
      </table>
    );

    expect(screen.getByText('John Doe')).toBeInTheDocument();
    expect(screen.getByText('#5')).toBeInTheDocument();
    expect(screen.getByText('85.5')).toBeInTheDocument();
    expect(screen.getByText('25.0%')).toBeInTheDocument();
    expect(screen.getByText('15.0%')).toBeInTheDocument();
    expect(screen.getByText('90.0%')).toBeInTheDocument();
    expect(screen.getByText('2 years')).toBeInTheDocument();
    expect(screen.getByTestId('probability-bar')).toBeInTheDocument();
  });

  it('applies correct CSS classes', () => {
    render(
      <table>
        <tbody>
          <BaseTableRow driver={mockDriver} />
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

  it('applies gradient background when empirical_percentage >= 50', () => {
    const highPercentageDriver = { ...mockDriver, empirical_percentage: 75 };

    render(
      <table>
        <tbody>
          <BaseTableRow driver={highPercentageDriver} />
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
          <BaseTableRow driver={lowPercentageDriver} />
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
          <BaseTableRow driver={nullPercentageDriver} />
        </tbody>
      </table>
    );

    expect(screen.getByTestId('probability-bar')).toHaveTextContent('0%');
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
});
