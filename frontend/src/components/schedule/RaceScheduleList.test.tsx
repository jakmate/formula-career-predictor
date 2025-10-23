import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { RaceScheduleList, type Race } from './RaceScheduleList';

// Mock lucide-react icons
vi.mock('lucide-react', () => ({
  Calendar: ({
    className,
    ...props
  }: {
    className?: string;
    [key: string]: unknown;
  }) => <div data-testid="calendar-icon" className={className} {...props} />,
  MapPin: ({
    className,
    ...props
  }: {
    className?: string;
    [key: string]: unknown;
  }) => <div data-testid="mappin-icon" className={className} {...props} />,
}));

// Mock the flags utility
vi.mock('../../utils/flags', () => ({
  getFlagComponent: vi.fn((location) => (
    <span data-testid="flag">{location}</span>
  )),
}));

describe('RaceScheduleList', () => {
  const mockRaces = [
    {
      slug: 'bahrain-gp',
      round: 1,
      name: 'Bahrain',
      location: 'Sakhir',
      sessions: {
        race: {
          start: '2024-03-02T15:00:00Z',
          time: '15:00',
        },
      },
    },
    {
      slug: 'saudi-gp',
      round: 2,
      name: 'Saudi Arabian',
      location: 'Jeddah',
      sessions: {
        race: {
          start: '2024-03-09T16:00:00Z',
          time: '16:00',
        },
      },
    },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
    // Mock current date to be before the races
    vi.setSystemTime(new Date('2024-02-01T12:00:00Z'));
  });

  // Helper to render with default series
  const renderWithSeries = (races: Race[], selectedSeries = 'f1') => {
    return render(
      <RaceScheduleList races={races} selectedSeries={selectedSeries} />
    );
  };

  it('renders empty state when no races provided', () => {
    renderWithSeries([]);

    expect(screen.getByTestId('calendar-icon')).toBeInTheDocument();
    expect(
      screen.getByText('No races found for this series')
    ).toBeInTheDocument();
  });

  it('renders races with "GRAND PRIX" for F1', () => {
    renderWithSeries(mockRaces, 'f1');

    expect(screen.getByText('Bahrain GRAND PRIX')).toBeInTheDocument();
    expect(screen.getByText('Saudi Arabian GRAND PRIX')).toBeInTheDocument();
  });

  it('renders races with "Grand Prix" for F2', () => {
    renderWithSeries(mockRaces, 'f2');

    expect(screen.getByText('Bahrain Grand Prix')).toBeInTheDocument();
    expect(screen.getByText('Saudi Arabian Grand Prix')).toBeInTheDocument();
  });

  it('renders races with "Grand Prix" for F3', () => {
    renderWithSeries(mockRaces, 'f3');

    expect(screen.getByText('Bahrain Grand Prix')).toBeInTheDocument();
    expect(screen.getByText('Saudi Arabian Grand Prix')).toBeInTheDocument();
  });

  it('displays timezone information', () => {
    renderWithSeries(mockRaces);

    const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    expect(
      screen.getByText(`Times shown in ${timezone.replace('_', ' ')}`)
    ).toBeInTheDocument();
  });

  it('formats dates and times correctly', () => {
    renderWithSeries(mockRaces);

    // The exact format depends on the system timezone, but we can check structure
    const dateElements = screen.getAllByText(/Mar \d+/);
    expect(dateElements.length).toBeGreaterThan(0);
  });

  it('handles TBC time correctly', () => {
    const racesWithTBC = [
      {
        ...mockRaces[0],
        sessions: {
          race: {
            start: '2024-03-02T15:00:00Z',
            time: 'TBC',
          },
        },
      },
    ];

    renderWithSeries(racesWithTBC);

    expect(screen.getByText('TBC')).toBeInTheDocument();
  });

  it('marks past races as completed', () => {
    // Set current date to be after the races
    vi.setSystemTime(new Date('2024-03-15T12:00:00Z'));

    renderWithSeries(mockRaces);

    expect(screen.getAllByText('COMPLETED')).toHaveLength(2);
  });

  it('marks next upcoming race correctly', () => {
    const futureMockRaces = [
      {
        slug: 'past-race',
        round: 1,
        name: 'Past Race',
        location: 'Past Location',
        sessions: {
          race: {
            start: '2024-01-01T15:00:00Z',
            time: '15:00',
          },
        },
      },
      {
        slug: 'next-race',
        round: 2,
        name: 'Next Race',
        location: 'Next Location',
        sessions: {
          race: {
            start: '2024-03-01T15:00:00Z',
            time: '15:00',
          },
        },
      },
      {
        slug: 'future-race',
        round: 3,
        name: 'Future Race',
        location: 'Future Location',
        sessions: {
          race: {
            start: '2024-04-01T15:00:00Z',
            time: '15:00',
          },
        },
      },
    ];

    renderWithSeries(futureMockRaces);

    expect(screen.getByText('COMPLETED')).toBeInTheDocument();
    expect(screen.getByText('NEXT RACE')).toBeInTheDocument();
    expect(screen.getAllByText('NEXT RACE')).toHaveLength(1);
  });

  it('handles races without start date', () => {
    const racesWithoutStart = [
      {
        slug: 'no-start',
        round: 1,
        name: 'No Start',
        location: 'Test Location',
        sessions: {
          race: {
            time: '15:00',
          },
        },
      },
    ];

    renderWithSeries(racesWithoutStart);

    expect(screen.getByText('No Start GRAND PRIX')).toBeInTheDocument();
  });

  it('handles date-only strings (10 characters)', () => {
    const racesWithDateOnly = [
      {
        slug: 'date-only',
        round: 1,
        name: 'Date Only',
        location: 'Test Location',
        sessions: {
          race: {
            start: '2024-03-02',
            time: '15:00',
          },
        },
      },
    ];

    renderWithSeries(racesWithDateOnly);

    expect(screen.getByText('Date Only GRAND PRIX')).toBeInTheDocument();
  });

  it('handles invalid date strings gracefully', () => {
    const racesWithInvalidDate = [
      {
        slug: 'invalid-date',
        round: 1,
        name: 'Invalid Date',
        location: 'Test Location',
        sessions: {
          race: {
            start: 'invalid-date',
            time: '15:00',
          },
        },
      },
    ];

    renderWithSeries(racesWithInvalidDate);

    expect(screen.getByText('Invalid Date GRAND PRIX')).toBeInTheDocument();
  });

  it('applies correct CSS classes for different race states', () => {
    const mixedRaces = [
      {
        slug: 'past-race',
        round: 1,
        name: 'Past Race',
        location: 'Past Location',
        sessions: {
          race: {
            start: '2024-01-01T15:00:00Z',
            time: '15:00',
          },
        },
      },
      {
        slug: 'next-race',
        round: 2,
        name: 'Next Race',
        location: 'Next Location',
        sessions: {
          race: {
            start: '2024-03-01T15:00:00Z',
            time: '15:00',
          },
        },
      },
    ];

    renderWithSeries(mixedRaces);

    const raceCards = screen
      .getAllByText(/GRAND PRIX/)
      .map((el) => el.closest('[class*="backdrop-blur-lg"]'));

    // Past race should have gray styling
    expect(raceCards[0]).toHaveClass('bg-gray-900/60');

    // Next race should have cyan/purple gradient
    expect(raceCards[1]).toHaveClass('bg-gradient-to-br');
  });

  it('uses slug as key when available, falls back to index', () => {
    const racesWithAndWithoutSlug = [
      {
        slug: 'with-slug',
        round: 1,
        name: 'With Slug',
        location: 'Location 1',
        sessions: { race: { start: '2024-03-02T15:00:00Z', time: '15:00' } },
      },
      {
        round: 2,
        name: 'Without Slug',
        location: 'Location 2',
        sessions: { race: { start: '2024-03-09T15:00:00Z', time: '15:00' } },
      },
    ];

    renderWithSeries(racesWithAndWithoutSlug);

    // Should render without errors
    expect(screen.getByText('With Slug GRAND PRIX')).toBeInTheDocument();
    expect(screen.getByText('Without Slug GRAND PRIX')).toBeInTheDocument();
  });
});
