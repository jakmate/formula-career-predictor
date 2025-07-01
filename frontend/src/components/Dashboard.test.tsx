import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import Dashboard from './Dashboard';
import { beforeEach, describe, expect, test, vi } from 'vitest';

vi.mock('./Navbar', () => ({
  Navbar: ({ activeView }: { activeView: string }) => (
    <div data-testid="navbar">{activeView}</div>
  ),
}));

vi.mock('./schedule/Schedule', () => ({
  Schedule: () => <div data-testid="schedule">Schedule</div>,
}));

vi.mock('./table/PredictionsTable', () => ({
  PredictionsTable: () => <div data-testid="predictions">Predictions</div>,
}));

const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

describe('Dashboard', () => {
  beforeEach(() => {
    mockNavigate.mockClear();
  });

  test('renders predictions view by default', () => {
    render(
      <MemoryRouter initialEntries={['/predictions']}>
        <Dashboard />
      </MemoryRouter>
    );

    expect(screen.getByTestId('predictions')).toBeInTheDocument();
    expect(screen.getByTestId('navbar')).toHaveTextContent('predictions');
  });

  test('renders schedule view', () => {
    render(
      <MemoryRouter initialEntries={['/schedule']}>
        <Dashboard />
      </MemoryRouter>
    );

    expect(screen.getByTestId('schedule')).toBeInTheDocument();
    expect(screen.getByTestId('navbar')).toHaveTextContent('schedule');
  });

  test('redirects from root path to predictions', () => {
    render(
      <MemoryRouter initialEntries={['/']}>
        <Dashboard />
      </MemoryRouter>
    );

    expect(mockNavigate).toHaveBeenCalledWith('/predictions', {
      replace: true,
    });
  });
});
