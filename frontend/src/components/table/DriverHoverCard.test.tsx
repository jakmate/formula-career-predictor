import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { DriverHoverCard } from './DriverHoverCard';
import type { Driver } from '../../types/Driver';

// Mock getFlagComponent
vi.mock('../../utils/flags', () => ({
  getFlagComponent: () => <div data-testid="flag" />,
}));

// Mock timer functions
beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.runOnlyPendingTimers();
  vi.useRealTimers();
});

const mockDriver: Driver = {
  driver: 'Lewis Hamilton',
  team: 'Mercedes',
  nationality: 'British',
  dob: '1985-01-07',
  age: 39,
  wins: 103,
  podiums: 197,
  position: 2,
  points: 214.5,
  experience: 16,
  win_rate: 0,
  top_10_rate: 0,
  dnf_rate: 0,
  participation_rate: 0,
  teammate_h2h: 0,
  team_pos: 0,
  team_points: 0,
};

describe('DriverHoverCard', () => {
  it('renders children', () => {
    render(
      <DriverHoverCard driver={mockDriver}>
        <button>Hover me</button>
      </DriverHoverCard>
    );

    expect(screen.getByText('Hover me')).toBeInTheDocument();
  });

  it('shows card on hover after delay', async () => {
    render(
      <DriverHoverCard driver={mockDriver}>
        <button>Trigger</button>
      </DriverHoverCard>
    );

    fireEvent.mouseEnter(screen.getByText('Trigger'));
    act(() => vi.advanceTimersByTime(300));

    expect(screen.getByText('Lewis Hamilton')).toBeVisible();
  });

  it('hides card on mouse leave', async () => {
    render(
      <DriverHoverCard driver={mockDriver}>
        <button>Trigger</button>
      </DriverHoverCard>
    );

    fireEvent.mouseEnter(screen.getByText('Trigger'));
    act(() => vi.advanceTimersByTime(300));
    fireEvent.mouseLeave(screen.getByText('Trigger'));

    expect(screen.queryByText('Lewis Hamilton')).not.toBeInTheDocument();
  });

  it('handles position calculation for top placement', () => {
    // Mock getBoundingClientRect for trigger element
    const mockRect = {
      bottom: 900,
      top: 800,
      left: 100,
      right: 200,
      width: 100,
      height: 50,
      x: 100,
      y: 800,
      toJSON: () => {},
    } as DOMRect;

    Element.prototype.getBoundingClientRect = vi.fn(() => mockRect);
    window.innerHeight = 1000;

    render(
      <DriverHoverCard driver={mockDriver}>
        <button>Trigger</button>
      </DriverHoverCard>
    );

    fireEvent.mouseEnter(screen.getByText('Trigger'));
    act(() => vi.advanceTimersByTime(300));

    // Find the card by its content instead of role
    const card = screen
      .getByText('Lewis Hamilton')
      .closest('div[class*="absolute"]');
    expect(card).toBeInTheDocument();
  });

  it('handles missing age field', () => {
    const driverWithoutAge = { ...mockDriver, age: undefined };
    render(
      <DriverHoverCard driver={driverWithoutAge}>
        <button>Trigger</button>
      </DriverHoverCard>
    );

    fireEvent.mouseEnter(screen.getByText('Trigger'));
    act(() => vi.advanceTimersByTime(300));

    expect(screen.getByText('Age:')).toBeInTheDocument();
    // Age should not be displayed when undefined
    const ageValue = screen.getByText('Age:').nextElementSibling;
    expect(ageValue).toHaveTextContent('');
  });

  it('displays experience correctly', () => {
    render(
      <DriverHoverCard driver={{ ...mockDriver, experience: 0 }}>
        <button>Trigger</button>
      </DriverHoverCard>
    );

    fireEvent.mouseEnter(screen.getByText('Trigger'));
    act(() => vi.advanceTimersByTime(300));

    expect(screen.getByText('Rookie')).toBeInTheDocument();
  });

  it('displays 1 year experience correctly', () => {
    render(
      <DriverHoverCard driver={{ ...mockDriver, experience: 1 }}>
        <button>Trigger</button>
      </DriverHoverCard>
    );

    fireEvent.mouseEnter(screen.getByText('Trigger'));
    act(() => vi.advanceTimersByTime(300));

    expect(screen.getByText('1 year')).toBeInTheDocument();
  });

  it('cancels show timer on mouse leave', () => {
    render(
      <DriverHoverCard driver={mockDriver}>
        <button>Trigger</button>
      </DriverHoverCard>
    );

    fireEvent.mouseEnter(screen.getByText('Trigger'));
    act(() => vi.advanceTimersByTime(100));
    fireEvent.mouseLeave(screen.getByText('Trigger'));
    act(() => vi.advanceTimersByTime(200));

    expect(screen.queryByText('Lewis Hamilton')).not.toBeInTheDocument();
  });
});
