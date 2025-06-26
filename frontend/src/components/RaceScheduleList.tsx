import { Calendar, MapPin } from "lucide-react";

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
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric'
    });
  };

  const isPastRace = (raceDate: string) => {
    if (!raceDate) return false;
    
    try {
      const now = new Date();
      const date = new Date(raceDate);
      
      // Handle TBC dates (date-only format)
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
    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
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
              backdrop-blur-lg rounded-lg border p-4 transition-opacity
              ${past 
                ? 'bg-red-900/40 border-red-800/50 opacity-60' 
                : isUpcoming 
                  ? 'bg-green-900/30 border-green-700/50' 
                  : 'bg-white/10 border-white/20'
              }
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
              </div>
              
              {raceDate && (
                <div className="mt-auto">
                  <div className="text-white font-medium">
                    {formatDate(raceDate)}
                  </div>
                  <div className="text-white/70 text-sm">
                    {isTBC ? 'TBC' : (
                      new Date(raceDate).toLocaleTimeString('en-US', {
                        hour: '2-digit',
                        minute: '2-digit'
                      })
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
};