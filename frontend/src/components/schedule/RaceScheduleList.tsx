import { Calendar, MapPin } from 'lucide-react';
import { getFlagComponent } from '../../utils/flags';

interface SessionDetails {
  start?: string;
  time?: string;
}

interface Session {
  race?: SessionDetails;
}

interface Race {
  slug?: string;
  round: number;
  name: string;
  location: string;
  sessions: Session;
}

interface RaceScheduleListProps {
  races: Race[];
}

export const RaceScheduleList = ({ races }: RaceScheduleListProps) => {
  const userTimezone = Intl.DateTimeFormat().resolvedOptions().timeZone;

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      timeZone: userTimezone,
      month: 'short',
      day: 'numeric',
    });
  };

  const formatTime = (dateString: string) => {
    return new Date(dateString).toLocaleTimeString('en-US', {
      timeZone: userTimezone,
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const isPastRace = (raceDate: string) => {
    if (!raceDate) return false;

    try {
      const now = new Date();
      const date = new Date(raceDate);

      if (raceDate.length === 10) {
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        return date < today;
      }
      return date < now;
    } catch {
      return false;
    }
  };

  let nextUpcomingFound = false;

  if (!races || races.length === 0) {
    return (
      <div className="text-center text-white/70 py-8">
        <Calendar className="w-12 h-12 mx-auto mb-4 text-white/50" />
        <p>No races found for this series</p>
      </div>
    );
  }

  return (
    <div className="space-y-2 mb-4">
      <div className="text-center text-white/50 text-sm">
        Times shown in {userTimezone.replace('_', ' ')}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {races.map((race: Race, index: number) => {
          const raceSession = race.sessions.race;
          const raceDate = raceSession?.start;
          const isTBC = raceSession?.time === 'TBC';
          const past = raceDate ? isPastRace(raceDate) : false;
          const isUpcoming = !past && !nextUpcomingFound;

          if (isUpcoming) nextUpcomingFound = true;

          return (
            <div
              key={race.slug || index}
              className={`
                backdrop-blur-lg rounded-xl border p-4 transition-all duration-200
                ${
                  past
                    ? 'bg-gray-900/60 border-gray-700 opacity-70'
                    : isUpcoming
                      ? 'bg-gradient-to-br from-cyan-900/30 to-purple-900/30 border-cyan-500/50 shadow-lg shadow-cyan-500/20'
                      : 'bg-gray-800/60 border-cyan-500/30'
                } hover:shadow-cyan-500/20 hover:border-cyan-400/50
              `}
            >
              <div className="flex flex-col h-full">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-white/70 text-sm font-medium">
                    Round {race.round}
                  </span>
                  <div>
                    {past && (
                      <span className="text-xs bg-red-800/70 text-white px-2 py-1 rounded">
                        COMPLETED
                      </span>
                    )}
                    {isUpcoming && (
                      <span className="text-xs bg-green-700 text-white px-2 py-1 rounded">
                        NEXT RACE
                      </span>
                    )}
                  </div>
                </div>

                <h3 className="text-white font-semibold text-lg mb-2">
                  {race.name} Grand Prix
                </h3>

                <div className="flex items-center mb-3 text-white/70">
                  <MapPin className="w-4 h-4 mr-1" />
                  <span className="text-sm">{race.location}</span>
                  <div className="ml-2">{getFlagComponent(race.location)}</div>
                </div>

                {raceDate && (
                  <div className="mt-auto">
                    <div className="text-white font-medium">
                      {formatDate(raceDate)}
                    </div>
                    <div className="text-white/70 text-sm">
                      {isTBC ? 'TBC' : formatTime(raceDate)}
                    </div>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
