import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { Schedule } from './Schedule';

// Mock child components
vi.mock('./NextRaceCard', () => ({
  NextRaceCard: ({ nextRace }: { nextRace: { name: string } | null }) => (
    <div data-testid="next-race-card">
      {nextRace ? `Next Race: ${nextRace.name}` : 'No next race'}
    </div>
  ),
}));

vi.mock('./RaceScheduleList', () => ({
  RaceScheduleList: ({ races }: { races: unknown[] }) => (
    <div data-testid="race-schedule-list">{races.length} races</div>
  ),
}));

vi.mock('../Header', () => ({
  Header: ({
    title,
    rightContent,
  }: {
    title: string;
    rightContent: React.ReactNode;
  }) => (
    <div data-testid="header">
      <h1>{title}</h1>
      <div data-testid="header-right-content">{rightContent}</div>
    </div>
  ),
}));

vi.mock('../ErrorDisplay', () => ({
  ErrorDisplay: ({ error }: { error: { message: string } }) => (
    <div data-testid="error-display">Error: {error.message}</div>
  ),
}));

// Mock the useSchedule hook
const mockUseSchedule = vi.fn();
vi.mock('../../hooks/useSchedule', () => ({
  useSchedule: () => mockUseSchedule(),
}));

// Mock Lucide icons
vi.mock('lucide-react', () => ({
  RefreshCw: ({ className }: { className?: string }) => (
    <div data-testid="refresh-icon" className={className} />
  ),
}));

const mockRaces = [
  {
    slug: 'bahrain-gp',
    round: 1,
    name: 'Bahrain',
    location: 'Bahrain',
    sessions: {
      race: {
        start: '2024-03-02T15:00:00Z',
        time: '15:00',
      },
    },
  },
  {
    slug: 'saudi-arabia-gp',
    round: 2,
    name: 'Saudi Arabia',
    location: 'Saudi Arabia',
    sessions: {
      race: {
        start: '2024-03-09T19:00:00Z',
        time: '19:00',
      },
    },
  },
];

const mockNextRace = {
  name: 'Monaco',
  round: 6,
  location: 'Monaco',
  sessions: {
    race: {
      start: '2024-05-26T15:00:00Z',
    },
  },
};

const mockSeries = [
  { value: 'f1', label: 'Formula 1' },
  { value: 'f2', label: 'Formula 2' },
];

describe('Schedule', () => {
  beforeEach(() => {
    mockUseSchedule.mockReturnValue({
      races: mockRaces,
      nextRace: mockNextRace,
      selectedSeries: 'f1',
      setSelectedSeries: vi.fn(),
      series: mockSeries,
      loading: false,
      error: null,
      refreshSchedule: vi.fn(),
    });
  });

  it('renders header with correct title', () => {
    render(<Schedule />);

    expect(screen.getByTestId('header')).toBeInTheDocument();
    expect(screen.getByText('Race Schedule')).toBeInTheDocument();
  });

  it('renders series selector with options', () => {
    render(<Schedule />);

    const select = screen.getByRole('combobox');
    expect(select).toBeInTheDocument();
    expect(select).toHaveValue('f1');

    expect(screen.getByText('Formula 1')).toBeInTheDocument();
    expect(screen.getByText('Formula 2')).toBeInTheDocument();
  });

  it('renders refresh button', () => {
    render(<Schedule />);

    const refreshButton = screen.getByText('Refresh');
    expect(refreshButton).toBeInTheDocument();
    expect(screen.getByTestId('refresh-icon')).toBeInTheDocument();
  });

  it('calls setSelectedSeries when series changes', () => {
    const mockSetSelectedSeries = vi.fn();
    mockUseSchedule.mockReturnValue({
      races: mockRaces,
      nextRace: mockNextRace,
      selectedSeries: 'f1',
      setSelectedSeries: mockSetSelectedSeries,
      series: mockSeries,
      loading: false,
      error: null,
      refetch: vi.fn(),
    });

    render(<Schedule />);

    const select = screen.getByRole('combobox');
    fireEvent.change(select, { target: { value: 'f2' } });

    expect(mockSetSelectedSeries).toHaveBeenCalledWith('f2');
  });

  it('calls refreshSchedule when refresh button is clicked', () => {
    const mockRefetch = vi.fn();
    mockUseSchedule.mockReturnValue({
      races: mockRaces,
      nextRace: mockNextRace,
      selectedSeries: 'f1',
      setSelectedSeries: vi.fn(),
      series: mockSeries,
      loading: false,
      error: null,
      refreshSchedule: mockRefetch,
    });

    render(<Schedule />);

    const refreshButton = screen.getByText('Refresh');
    fireEvent.click(refreshButton);

    expect(mockRefetch).toHaveBeenCalled();
  });

  it('renders NextRaceCard when nextRace exists', () => {
    render(<Schedule />);

    expect(screen.getByTestId('next-race-card')).toBeInTheDocument();
    expect(screen.getByText('Next Race: Monaco')).toBeInTheDocument();
  });

  it('does not render NextRaceCard when nextRace is null', () => {
    mockUseSchedule.mockReturnValue({
      races: mockRaces,
      nextRace: null,
      selectedSeries: 'f1',
      setSelectedSeries: vi.fn(),
      series: mockSeries,
      loading: false,
      error: null,
      refreshSchedule: vi.fn(),
    });

    render(<Schedule />);

    expect(screen.queryByTestId('next-race-card')).not.toBeInTheDocument();
  });

  it('renders error display when error exists', () => {
    const mockError = { message: 'Failed to fetch data' };
    mockUseSchedule.mockReturnValue({
      races: mockRaces,
      nextRace: mockNextRace,
      selectedSeries: 'f1',
      setSelectedSeries: vi.fn(),
      series: mockSeries,
      loading: false,
      error: mockError,
      refreshSchedule: vi.fn(),
    });

    render(<Schedule />);

    expect(screen.getByTestId('error-display')).toBeInTheDocument();
    expect(screen.getByText('Error: Failed to fetch data')).toBeInTheDocument();
  });

  it('renders loading state correctly', () => {
    mockUseSchedule.mockReturnValue({
      races: mockRaces,
      nextRace: mockNextRace,
      selectedSeries: 'f1',
      setSelectedSeries: vi.fn(),
      series: mockSeries,
      loading: true,
      error: null,
      refreshSchedule: vi.fn(),
    });

    render(<Schedule />);

    expect(screen.getByText('Loading schedule...')).toBeInTheDocument();
    expect(screen.getByText('Updating...')).toBeInTheDocument();

    // Loading spinner should be present
    const loadingSpinner = screen.getAllByTestId('refresh-icon')[0];
    expect(loadingSpinner).toHaveClass('animate-spin');
  });

  it('disables refresh button when loading', () => {
    mockUseSchedule.mockReturnValue({
      races: mockRaces,
      nextRace: mockNextRace,
      selectedSeries: 'f1',
      setSelectedSeries: vi.fn(),
      series: mockSeries,
      loading: true,
      error: null,
      refreshSchedule: vi.fn(),
    });

    render(<Schedule />);

    const refreshButton = screen.getByText('Updating...');
    expect(refreshButton).toBeDisabled();
  });

  it('renders RaceScheduleList when not loading', () => {
    render(<Schedule />);

    expect(screen.getByTestId('race-schedule-list')).toBeInTheDocument();
    expect(screen.getByText('2 races')).toBeInTheDocument();
  });

  it('renders Full Season Schedule header', () => {
    render(<Schedule />);

    expect(screen.getByText('Full Season Schedule')).toBeInTheDocument();
  });

  it('handles empty races array', () => {
    mockUseSchedule.mockReturnValue({
      races: [],
      nextRace: null,
      selectedSeries: 'f1',
      setSelectedSeries: vi.fn(),
      series: mockSeries,
      loading: false,
      error: null,
      refreshSchedule: vi.fn(),
    });

    render(<Schedule />);

    expect(screen.getByText('0 races')).toBeInTheDocument();
  });

  it('handles empty series array', () => {
    mockUseSchedule.mockReturnValue({
      races: mockRaces,
      nextRace: mockNextRace,
      selectedSeries: 'f1',
      setSelectedSeries: vi.fn(),
      series: [],
      loading: false,
      error: null,
      refreshSchedule: vi.fn(),
    });

    render(<Schedule />);

    const select = screen.getByRole('combobox');
    expect(select).toBeInTheDocument();
    expect(select.children).toHaveLength(0);
  });

  it('applies correct styling classes', () => {
    render(<Schedule />);

    const select = screen.getByRole('combobox');
    expect(select).toHaveClass(
      'px-4',
      'py-2',
      'bg-gray-800/60',
      'border',
      'border-cyan-500/30'
    );

    const refreshButton = screen.getByText('Refresh');
    expect(refreshButton).toHaveClass(
      'px-6',
      'py-2',
      'bg-gradient-to-r',
      'from-cyan-600',
      'to-purple-600'
    );
  });

  it('renders refresh icon with correct animation class when loading', () => {
    mockUseSchedule.mockReturnValue({
      races: mockRaces,
      nextRace: mockNextRace,
      selectedSeries: 'f1',
      setSelectedSeries: vi.fn(),
      series: mockSeries,
      loading: true,
      error: null,
      refreshSchedule: vi.fn(),
    });

    render(<Schedule />);

    const refreshIcons = screen.getAllByTestId('refresh-icon');
    expect(refreshIcons[0]).toHaveClass('animate-spin');
  });
});
