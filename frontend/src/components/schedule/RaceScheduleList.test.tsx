import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { RaceScheduleList } from './RaceScheduleList';

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

  it('renders empty state when no races provided', () => {
    render(<RaceScheduleList races={[]} />);

    expect(screen.getByTestId('calendar-icon')).toBeInTheDocument();
    expect(
      screen.getByText('No races found for this series')
    ).toBeInTheDocument();
  });

  it('renders races correctly', () => {
    render(<RaceScheduleList races={mockRaces} />);

    expect(screen.getByText('Bahrain Grand Prix')).toBeInTheDocument();
    expect(screen.getByText('Saudi Arabian Grand Prix')).toBeInTheDocument();
    expect(screen.getByText('Round 1')).toBeInTheDocument();
    expect(screen.getByText('Round 2')).toBeInTheDocument();
  });

  it('displays timezone information', () => {
    render(<RaceScheduleList races={mockRaces} />);

    const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    expect(
      screen.getByText(`Times shown in ${timezone.replace('_', ' ')}`)
    ).toBeInTheDocument();
  });

  it('formats dates and times correctly', () => {
    render(<RaceScheduleList races={mockRaces} />);

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

    render(<RaceScheduleList races={racesWithTBC} />);

    expect(screen.getByText('TBC')).toBeInTheDocument();
  });

  it('marks past races as completed', () => {
    // Set current date to be after the races
    vi.setSystemTime(new Date('2024-03-15T12:00:00Z'));

    render(<RaceScheduleList races={mockRaces} />);

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

    render(<RaceScheduleList races={futureMockRaces} />);

    expect(screen.getByText('COMPLETED')).toBeInTheDocument();
    expect(screen.getByText('NEXT RACE')).toBeInTheDocument();
    // Should only have one "NEXT RACE" label
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

    render(<RaceScheduleList races={racesWithoutStart} />);

    expect(screen.getByText('No Start Grand Prix')).toBeInTheDocument();
    // Should not crash and should render the race
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

    render(<RaceScheduleList races={racesWithDateOnly} />);

    expect(screen.getByText('Date Only Grand Prix')).toBeInTheDocument();
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

    render(<RaceScheduleList races={racesWithInvalidDate} />);

    expect(screen.getByText('Invalid Date Grand Prix')).toBeInTheDocument();
    // Should not crash
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

    render(<RaceScheduleList races={mixedRaces} />);

    const raceCards = screen
      .getAllByText(/Grand Prix/)
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

    render(<RaceScheduleList races={racesWithAndWithoutSlug} />);

    // Should render without errors
    expect(screen.getByText('With Slug Grand Prix')).toBeInTheDocument();
    expect(screen.getByText('Without Slug Grand Prix')).toBeInTheDocument();
  });
});
