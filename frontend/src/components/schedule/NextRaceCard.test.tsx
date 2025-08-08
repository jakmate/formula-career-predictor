import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, act, within } from '@testing-library/react';
import { NextRaceCard } from './NextRaceCard';

// Mock the flags utility
vi.mock('../../utils/flags', () => ({
  getFlagComponent: vi.fn((location: string) => (
    <span data-testid="flag">{location}</span>
  )),
}));

// Mock Lucide icons
vi.mock('lucide-react', () => ({
  Clock: ({ className }: { className?: string }) => (
    <div data-testid="clock-icon" className={className} />
  ),
  MapPin: ({ className }: { className?: string }) => (
    <div data-testid="mappin-icon" className={className} />
  ),
  Calendar: ({ className }: { className?: string }) => (
    <div data-testid="calendar-icon" className={className} />
  ),
  Trophy: ({ className }: { className?: string }) => (
    <div data-testid="trophy-icon" className={className} />
  ),
}));

const mockNextRace = {
  name: 'Monaco',
  round: 6,
  totalRounds: 24,
  location: 'Monaco',
  sessions: {
    fp1: {
      start: '2024-05-24T13:30:00Z',
      end: '2024-05-24T14:30:00Z',
    },
    fp2: {
      start: '2024-05-24T17:00:00Z',
      end: '2024-05-24T18:00:00Z',
    },
    fp3: {
      start: '2024-05-25T12:30:00Z',
      end: '2024-05-25T13:30:00Z',
    },
    qualifying: {
      start: '2024-05-25T16:00:00Z',
      end: '2024-05-25T17:00:00Z',
    },
    race: {
      start: '2024-05-26T15:00:00Z',
      end: '2024-05-26T17:00:00Z',
    },
  },
  nextSession: {
    name: 'practice 1',
    date: '2024-05-24T13:30:00Z',
  },
};

describe('NextRaceCard', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2024-05-24T10:00:00Z'));
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('renders null when nextRace is null', () => {
    const { container } = render(<NextRaceCard nextRace={null} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders race information correctly', () => {
    render(<NextRaceCard nextRace={mockNextRace} />);

    expect(screen.getByText('NEXT RACE: Monaco GP')).toBeInTheDocument();
    expect(screen.getByText('Round 6 of 24')).toBeInTheDocument();
    const monacoElements = screen.getAllByText('Monaco');
    expect(monacoElements.length).toBeGreaterThan(0);

    expect(screen.getByTestId('flag')).toBeInTheDocument();
  });

  it('displays session timetable with correct sessions', () => {
    render(<NextRaceCard nextRace={mockNextRace} />);

    expect(screen.getByText('PRACTICE 1')).toBeInTheDocument();
    expect(screen.getByText('PRACTICE 2')).toBeInTheDocument();
    expect(screen.getByText('PRACTICE 3')).toBeInTheDocument();
    expect(screen.getByText('QUALIFYING')).toBeInTheDocument();
    expect(screen.getByText('RACE')).toBeInTheDocument();
  });

  it('handles TBC session times', () => {
    const raceWithTBC = {
      ...mockNextRace,
      sessions: {
        ...mockNextRace.sessions,
        fp1: {
          start: '2024-05-24T13:30:00Z',
          time: 'TBC',
        },
      },
    };

    render(<NextRaceCard nextRace={raceWithTBC} />);
    expect(screen.getByText('TBC')).toBeInTheDocument();
  });

  it('updates countdown timer', () => {
    render(<NextRaceCard nextRace={mockNextRace} />);

    // Initially shows hours and minutes
    expect(screen.getByText(/3h 30m/)).toBeInTheDocument();

    // Advance time by 1 hour
    act(() => {
      vi.advanceTimersByTime(60 * 60 * 1000);
    });

    expect(screen.getByText(/2h 30m/)).toBeInTheDocument();
  });

  it('shows "LIVE NOW" when session is active', () => {
    // Set time to during the session
    vi.setSystemTime(new Date('2024-05-24T14:00:00Z'));

    render(<NextRaceCard nextRace={mockNextRace} />);

    const liveNowElements = screen.getAllByText('LIVE NOW');
    expect(liveNowElements.length).toBeGreaterThan(0);
  });

  it('displays session status correctly', () => {
    // Set time to after some sessions
    vi.setSystemTime(new Date('2024-05-25T10:00:00Z'));

    render(<NextRaceCard nextRace={mockNextRace} />);

    // FP1 and FP2 should be completed
    const completedTexts = screen.getAllByText('COMPLETED');
    expect(completedTexts).toHaveLength(2);

    // FP3 should be upcoming
    const upcomingTexts = screen.getAllByText('UPCOMING');
    expect(upcomingTexts.length).toBeGreaterThan(0);
  });

  it('formats dates correctly', () => {
    render(<NextRaceCard nextRace={mockNextRace} />);

    expect(screen.getByText('Sunday, 26 May, 2024')).toBeInTheDocument();
  });

  it('displays user timezone when provided', () => {
    render(
      <NextRaceCard nextRace={mockNextRace} userTimezone="America/New_York" />
    );

    expect(
      screen.getByText('All times in America/New York')
    ).toBeInTheDocument();
  });

  it('shows default timezone when not provided', () => {
    render(<NextRaceCard nextRace={mockNextRace} />);

    expect(screen.getByText('All times in Local Time')).toBeInTheDocument();
  });

  it('renders without nextSession', () => {
    const raceWithoutNextSession = {
      ...mockNextRace,
      nextSession: undefined,
    };

    render(<NextRaceCard nextRace={raceWithoutNextSession} />);

    expect(screen.getByText('NEXT RACE: Monaco GP')).toBeInTheDocument();
    expect(screen.queryByText('Next Session')).not.toBeInTheDocument();
  });

  it('handles countdown with days', () => {
    // Set time to show days in countdown
    vi.setSystemTime(new Date('2024-05-22T10:00:00Z'));

    render(<NextRaceCard nextRace={mockNextRace} />);

    expect(screen.getByText(/2d 3h/)).toBeInTheDocument();
  });

  it('handles countdown with only minutes and seconds', () => {
    // Set time to show only minutes and seconds
    vi.setSystemTime(new Date('2024-05-24T13:25:00Z'));

    render(<NextRaceCard nextRace={mockNextRace} />);

    expect(screen.getByText(/5m 0s/)).toBeInTheDocument();
  });

  it('renders all required icons', () => {
    render(<NextRaceCard nextRace={mockNextRace} />);

    expect(screen.getByTestId('trophy-icon')).toBeInTheDocument();
    expect(screen.getByTestId('mappin-icon')).toBeInTheDocument();
    expect(screen.getByTestId('calendar-icon')).toBeInTheDocument();
    expect(screen.getByTestId('clock-icon')).toBeInTheDocument();
  });

  it('applies correct styling for live sessions', () => {
    vi.setSystemTime(new Date('2024-05-24T14:00:00Z'));

    render(<NextRaceCard nextRace={mockNextRace} />);

    const liveElements = screen.getAllByText('LIVE NOW');
    expect(liveElements).toHaveLength(2);
    expect(liveElements[0]).toHaveClass('text-red-400');
    expect(liveElements[1]).toHaveClass('text-red-400');
  });

  it('handles sessions with sprint qualifying and sprint', () => {
    const sprintRace = {
      ...mockNextRace,
      sessions: {
        ...mockNextRace.sessions,
        sprint_qualifying: {
          start: '2024-05-25T15:00:00Z',
          end: '2024-05-25T16:00:00Z',
        },
        sprint: {
          start: '2024-05-26T11:00:00Z',
          end: '2024-05-26T12:00:00Z',
        },
      },
    };

    render(<NextRaceCard nextRace={sprintRace} />);

    expect(screen.getByText('SPRINT QUALIFYING')).toBeInTheDocument();
    expect(screen.getByText('SPRINT')).toBeInTheDocument();
  });

  it('handles missing totalRounds', () => {
    const raceWithoutTotalRounds = {
      ...mockNextRace,
      totalRounds: undefined,
    };

    render(<NextRaceCard nextRace={raceWithoutTotalRounds} />);

    expect(screen.getByText('Round 6 of ?')).toBeInTheDocument();
  });

  it('handles YYYY-MM-DD date format correctly', () => {
    const raceWithSimpleDate = {
      ...mockNextRace,
      sessions: {
        ...mockNextRace.sessions,
        race: {
          start: '2024-05-26',
        },
      },
    };

    render(<NextRaceCard nextRace={raceWithSimpleDate} />);
    expect(screen.getByText('Sunday, 26 May, 2024')).toBeInTheDocument();
  });

  it('handles invalid time strings', () => {
    const raceWithInvalidTime = {
      ...mockNextRace,
      sessions: {
        ...mockNextRace.sessions,
        fp1: {
          start: 'invalid-time',
        },
      },
    };

    render(<NextRaceCard nextRace={raceWithInvalidTime} />);

    const practice1Session = screen
      .getByText('PRACTICE 1')
      .closest('.rounded-xl') as HTMLElement;

    // Target specific element by class
    const timeElement = within(practice1Session).getByText('Invalid Date', {
      selector: '.text-xl',
    });
    expect(timeElement).toBeInTheDocument();
  });

  it('handles YYYY-MM-DD format in short dates', () => {
    const raceWithSimpleDate = {
      ...mockNextRace,
      sessions: {
        ...mockNextRace.sessions,
        fp1: {
          start: '2024-05-24',
        },
      },
    };

    render(<NextRaceCard nextRace={raceWithSimpleDate} />);

    // Find date text specifically in the PRACTICE 1 session
    const practice1Session = screen
      .getByText('PRACTICE 1')
      .closest('.rounded-xl') as HTMLElement;
    const dateElement = within(practice1Session).getByText('Fri, 24 May');
    expect(dateElement).toBeInTheDocument();
  });

  describe('session status detection', () => {
    it('returns "tbc" for invalid session times', () => {
      const raceWithInvalidTime = {
        ...mockNextRace,
        sessions: {
          ...mockNextRace.sessions,
          fp1: {
            start: 'invalid-time',
          },
        },
      };

      render(<NextRaceCard nextRace={raceWithInvalidTime} />);

      const practice1Session = screen
        .getByText('PRACTICE 1')
        .closest('.rounded-xl') as HTMLElement;
      const withinSession = within(practice1Session);

      // Verify status texts
      expect(withinSession.queryByText('UPCOMING')).not.toBeInTheDocument();
      expect(withinSession.queryByText('LIVE NOW')).not.toBeInTheDocument();
      expect(withinSession.getByText('COMPLETED')).toBeInTheDocument();

      // Verify time is shown as TBC in both places
      const invalidDates = withinSession.getAllByText('TBC');
      expect(invalidDates).toHaveLength(1);
    });

    it('returns "upcoming" for future sessions', () => {
      vi.setSystemTime(new Date('2024-05-24T09:00:00Z'));
      render(<NextRaceCard nextRace={mockNextRace} />);
      expect(screen.getAllByText('UPCOMING').length).toBe(5);
    });

    it('returns "live" for ongoing sessions', () => {
      vi.setSystemTime(new Date('2024-05-24T13:45:00Z'));
      render(<NextRaceCard nextRace={mockNextRace} />);
      expect(screen.getAllByText('LIVE NOW').length).toBe(2);
    });
  });

  it('replaces underscores in timezone names', () => {
    render(
      <NextRaceCard nextRace={mockNextRace} userTimezone="America/New_York" />
    );
    expect(
      screen.getByText('All times in America/New York')
    ).toBeInTheDocument();
  });

  it('displays session timetable header correctly', () => {
    render(<NextRaceCard nextRace={mockNextRace} />);
    expect(screen.getByText('SESSION TIMETABLE')).toBeInTheDocument();
    expect(screen.getByText('All times in Local Time')).toBeInTheDocument();
  });
});
