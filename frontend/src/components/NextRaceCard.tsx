import { Clock, MapPin, Calendar, Trophy } from "lucide-react";

interface SessionTimes {
  [key: string]: string;
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
  sessions: SessionTimes;
  nextSession?: NextSession;
}

interface NextRaceCardProps {
  nextRace: NextRace | null;
}

export const NextRaceCard = ({ nextRace }: NextRaceCardProps) => {
  if (!nextRace) return null;

  // Format date in user's locale
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString(undefined, {
      weekday: 'long',
      month: 'long',
      day: 'numeric',
      year: 'numeric'
    });
  };

  // Format time in user's locale
  const formatTime = (dateString: string) => {
    return new Date(dateString).toLocaleTimeString(undefined, {
      hour: '2-digit',
      minute: '2-digit',
      timeZoneName: 'short'
    });
  };

  // Format time only (without date)
  const formatTimeOnly = (dateString: string) => {
    return new Date(dateString).toLocaleTimeString(undefined, {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div className="bg-gradient-to-r from-blue-700/30 via-purple-700/30 to-indigo-700/30 border border-blue-500/40 rounded-2xl p-8 mb-8 shadow-lg shadow-blue-500/20 relative overflow-hidden">
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-yellow-500 to-red-500"></div>
      <div className="absolute -top-4 -right-4 w-24 h-24 bg-yellow-400/10 rounded-full blur-xl"></div>
      <div className="absolute -bottom-4 -left-4 w-32 h-32 bg-blue-400/10 rounded-full blur-xl"></div>
      
      <div className="flex flex-col items-center text-center mb-8">
        <div className="flex items-center mb-3">
          <Trophy className="w-8 h-8 mr-3 text-yellow-400" />
          <h2 className="text-2xl md:text-3xl font-bold text-white">
            NEXT RACE: {nextRace.name} GP
          </h2>
        </div>
        
        <div className="bg-blue-500/30 px-4 py-1.5 rounded-full flex items-center">
          <span className="text-blue-200 text-sm font-medium">
            Round {nextRace.round} of {nextRace.totalRounds || "?"}
          </span>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8 justify-items-center">
        <div className="flex flex-col items-center max-w-xs text-center">
          <div className="flex items-center mb-2">
            <MapPin className="w-6 h-6 mr-2 text-red-400" />
            <h3 className="text-white/80 text-sm font-medium">Location</h3>
          </div>
          <p className="text-white text-xl font-semibold">{nextRace.location}</p>
        </div>
        
        <div className="flex flex-col items-center max-w-xs text-center">
          <div className="flex items-center mb-2">
            <Calendar className="w-6 h-6 mr-2 text-green-400" />
            <h3 className="text-white/80 text-sm font-medium">Race Day</h3>
          </div>
          <p className="text-white text-xl font-semibold">
            {formatDate(nextRace.sessions.race)}
          </p>
        </div>
        
        {nextRace.nextSession && (
          <div className="flex flex-col items-center max-w-xs text-center">
            <div className="flex items-center mb-2">
              <Clock className="w-6 h-6 mr-2 text-cyan-400" />
              <h3 className="text-white/80 text-sm font-medium">Next Session</h3>
            </div>
            <p className="text-white text-xl font-semibold">
              {nextRace.nextSession.name.toUpperCase()} - {formatTime(nextRace.nextSession.date)}
            </p>
          </div>
        )}
      </div>

      {nextRace.sessions && (
        <div className="bg-black/20 rounded-xl p-6 border border-white/10">
          <h3 className="text-white/80 text-sm font-medium mb-5 text-center uppercase tracking-wider">
            SESSION TIMETABLE
          </h3>
          
          <div className="flex flex-wrap justify-center gap-4">
            {Object.entries(nextRace.sessions).map(([session, time]) => (
              <div 
                key={session} 
                className="bg-gradient-to-b from-white/10 to-white/5 rounded-xl p-5 border border-white/10 flex flex-col items-center min-w-[160px]"
              >
                <div className="text-lg text-white font-bold uppercase mb-1">
                  {session === 'gp' ? 'RACE' : session.replace('fp', 'FP')}
                </div>
                <div className="text-white text-2xl font-bold mb-1">
                  {formatTimeOnly(time)}
                </div>
                <div className="text-white/70 text-sm">
                  {new Date(time).toLocaleDateString(undefined, {
                    weekday: 'short',
                    month: 'short',
                    day: 'numeric'
                  })}
                </div>
                <div className="text-white/60 text-xs mt-1">
                  {new Date(time).toLocaleTimeString(undefined, {
                    timeZoneName: 'short'
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};