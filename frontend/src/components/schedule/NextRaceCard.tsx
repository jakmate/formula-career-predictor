import { Clock, MapPin, Calendar, Trophy } from 'lucide-react';
import { useEffect, useState } from 'react';

const sessionDisplayNames: Record<string, string> = {
  practice: 'PRACTICE',
  fp1: 'PRACTICE 1',
  fp2: 'PRACTICE 2',
  fp3: 'PRACTICE 3',
  qualifying: 'QUALIFYING',
  sprint_qualifying: 'SPRINT QUALIFYING',
  sprint: 'SPRINT',
  race: 'RACE',
};

interface SessionInfo {
  start: string;
  end?: string;
  time?: string;
}

interface NextSession {
  name: string;
  date: string;
}

interface NextRace {
  name: string;
  round: number;
  totalRounds?: number;
  location: string;
  sessions: Record<string, SessionInfo>;
  nextSession?: NextSession;
}

interface NextRaceCardProps {
  nextRace: NextRace | null;
  userTimezone?: string;
}

export const NextRaceCard = ({ nextRace, userTimezone }: NextRaceCardProps) => {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [countdown, setCountdown] = useState<string | null>(null);

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (nextRace?.nextSession) {
      const updateCountdown = () => {
        const sessionTime = new Date(nextRace.nextSession!.date);
        const diff = sessionTime.getTime() - currentTime.getTime();

        if (diff <= 0) {
          setCountdown('LIVE NOW');
          return;
        }

        const days = Math.floor(diff / (1000 * 60 * 60 * 24));
        const hours = Math.floor(
          (diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60)
        );
        const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
        const seconds = Math.floor((diff % (1000 * 60)) / 1000);

        if (days > 0) {
          setCountdown(`${days}d ${hours}h`);
        } else if (hours > 0) {
          setCountdown(`${hours}h ${minutes}m`);
        } else {
          setCountdown(`${minutes}m ${seconds}s`);
        }
      };

      updateCountdown();
    }
  }, [currentTime, nextRace]);

  if (!nextRace) return null;

  // Format date - times are already converted by backend
  const formatDate = (dateString: string) => {
    if (dateString.length === 10) {
      const [year, month, day] = dateString.split('-');
      return new Date(
        parseInt(year),
        parseInt(month) - 1,
        parseInt(day)
      ).toLocaleDateString(undefined, {
        weekday: 'long',
        month: 'long',
        day: 'numeric',
        year: 'numeric',
      });
    }

    try {
      return new Date(dateString).toLocaleDateString(undefined, {
        weekday: 'long',
        month: 'long',
        day: 'numeric',
        year: 'numeric',
      });
    } catch {
      return 'Date TBC';
    }
  };

  // Format time - backend already converted to local timezone
  const formatTime = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleTimeString(undefined, {
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return 'TBC';
    }
  };

  // Format short date
  const formatShortDate = (dateString: string) => {
    if (dateString.length === 10) {
      const [year, month, day] = dateString.split('-');
      return new Date(
        parseInt(year),
        parseInt(month) - 1,
        parseInt(day)
      ).toLocaleDateString(undefined, {
        weekday: 'short',
        month: 'short',
        day: 'numeric',
      });
    }

    try {
      return new Date(dateString).toLocaleDateString(undefined, {
        weekday: 'short',
        month: 'short',
        day: 'numeric',
      });
    } catch {
      return 'TBC';
    }
  };

  // Determine session status
  const getSessionStatus = (sessionInfo: SessionInfo) => {
    if (sessionInfo.time === 'TBC') return 'tbc';

    try {
      const start = new Date(sessionInfo.start);
      const end = sessionInfo.end ? new Date(sessionInfo.end) : null;

      if (currentTime < start) return 'upcoming';
      if (end && currentTime <= end) return 'live';
      return 'completed';
    } catch {
      return 'tbc';
    }
  };

  const displayTimezone = userTimezone?.replace('_', ' ') || 'Local Time';

  return (
    <div className="bg-gradient-to-r from-cyan-900/20 via-purple-900/20 to-cyan-900/20 border border-cyan-500/30 rounded-2xl p-6 mb-8 shadow-xl shadow-cyan-500/10 relative overflow-hidden">
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-cyan-400 to-purple-500"></div>
      <div className="absolute -top-4 -right-4 w-24 h-24 bg-yellow-400/10 rounded-full blur-xl"></div>
      <div className="absolute -bottom-4 -left-4 w-32 h-32 bg-blue-400/10 rounded-full blur-xl"></div>

      <div className="flex flex-col items-center text-center mb-8 relative z-10">
        <div className="flex items-center mb-3">
          <div className="relative mr-3">
            <Trophy className="w-8 h-8 text-amber-300 z-10 relative" />
            <div className="absolute inset-0 bg-amber-400 rounded-full blur-sm opacity-40"></div>
          </div>
          <h2 className="text-2xl md:text-3xl font-bold text-white">
            NEXT RACE: {nextRace.name} GP
          </h2>
        </div>

        <div className="bg-cyan-900/40 px-4 py-1.5 rounded-full flex items-center border border-cyan-500/30">
          <span className="text-cyan-300 text-sm font-medium">
            Round {nextRace.round} of {nextRace.totalRounds || '?'}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8 justify-items-center">
        <div className="flex flex-col items-center max-w-xs text-center">
          <div className="flex items-center mb-2">
            <MapPin className="w-6 h-6 mr-2 text-red-400" />
            <h3 className="text-white/80 text-sm font-medium">Location</h3>
          </div>
          <p className="text-white text-xl font-semibold">
            {nextRace.location}
          </p>
        </div>

        <div className="flex flex-col items-center max-w-xs text-center">
          <div className="flex items-center mb-2">
            <Calendar className="w-6 h-6 mr-2 text-green-400" />
            <h3 className="text-white/80 text-sm font-medium">Race Day</h3>
          </div>
          <p className="text-white text-xl font-semibold">
            {formatDate(nextRace.sessions.race.start)}
          </p>
        </div>

        {nextRace.nextSession && (
          <div className="flex flex-col items-center max-w-xs text-center">
            <div className="flex items-center mb-2">
              <Clock className="w-6 h-6 mr-2 text-cyan-400" />
              <h3 className="text-white/80 text-sm font-medium">
                Next Session
              </h3>
            </div>
            <p className="text-white text-xl font-semibold">
              {nextRace.nextSession.name.toUpperCase()} -
              {countdown ? (
                <span
                  className={`inline-block ml-1 font-bold ${
                    countdown === 'LIVE NOW'
                      ? 'text-red-400 animate-pulse'
                      : 'text-cyan-300'
                  }`}
                >
                  {countdown}
                </span>
              ) : (
                <span className="inline-block text-cyan-300 ml-1 font-bold">
                  Loading...
                </span>
              )}
            </p>
          </div>
        )}
      </div>

      {nextRace.sessions && (
        <div className="bg-black/20 rounded-xl p-6 border border-white/10">
          <div className="flex flex-col items-center mb-5">
            <h3 className="text-white/80 text-sm font-medium mb-1 text-center uppercase tracking-wider">
              SESSION TIMETABLE
            </h3>
            <div className="text-white/50 text-xs">
              All times in {displayTimezone}
            </div>
          </div>

          <div className="flex flex-wrap justify-center gap-4">
            {Object.entries(nextRace.sessions).map(
              ([sessionKey, sessionInfo]) => {
                const sessionName =
                  sessionDisplayNames[sessionKey] || sessionKey.toUpperCase();
                const isTBC = sessionInfo.time === 'TBC';
                const status = getSessionStatus(sessionInfo);

                let sessionStyle =
                  'bg-gradient-to-b from-white/10 to-white/5 border-white/10';
                let glowStyle = '';

                if (status === 'live') {
                  sessionStyle =
                    'bg-gradient-to-b from-red-900/30 to-red-800/20 border-red-500';
                  glowStyle = 'shadow-lg shadow-red-500/30 animate-pulse';
                } else if (status === 'upcoming') {
                  sessionStyle =
                    'bg-gradient-to-b from-green-900/30 to-green-800/20 border-green-500';
                  glowStyle = 'shadow-lg shadow-green-500/30';
                } else if (status === 'completed') {
                  sessionStyle =
                    'bg-gradient-to-b from-gray-900/30 to-gray-800/20 border-gray-700 opacity-70';
                }

                return (
                  <div
                    key={sessionKey}
                    className={`rounded-xl p-5 border flex flex-col items-center min-w-[160px] ${sessionStyle} ${glowStyle}`}
                  >
                    <div className="text-lg text-white font-bold uppercase mb-1">
                      {sessionName}
                    </div>

                    {isTBC ? (
                      <div className="text-white text-xl font-bold mb-1">
                        TBC
                      </div>
                    ) : (
                      <>
                        <div className="flex flex-col items-center">
                          <div className="text-white text-xl font-bold">
                            {formatTime(sessionInfo.start)}
                          </div>
                          {sessionInfo.end && (
                            <>
                              <div className="text-white/70 text-sm">to</div>
                              <div className="text-white text-xl font-bold">
                                {formatTime(sessionInfo.end)}
                              </div>
                            </>
                          )}
                        </div>

                        <div className="mt-2 text-center">
                          <div className="text-white/70 text-sm">
                            {formatShortDate(sessionInfo.start)}
                          </div>
                          {status === 'live' && (
                            <div className="mt-1 text-xs font-bold text-red-400">
                              LIVE NOW
                            </div>
                          )}
                          {status === 'upcoming' && (
                            <div className="mt-1 text-xs font-bold text-green-400">
                              UPCOMING
                            </div>
                          )}
                          {status === 'completed' && (
                            <div className="mt-1 text-xs font-bold text-gray-400">
                              COMPLETED
                            </div>
                          )}
                        </div>
                      </>
                    )}
                  </div>
                );
              }
            )}
          </div>
        </div>
      )}
    </div>
  );
};
