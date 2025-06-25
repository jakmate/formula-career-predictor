import { RefreshCw } from "lucide-react";
import { NextRaceCard } from "./NextRaceCard";
import { RaceScheduleList } from "./RaceScheduleList";
import { useSchedule } from "../hooks/useSchedule";
import { Header } from "./Header";
import { ErrorDisplay } from "./ErrorDisplay";

export const Schedule = () => {
  const {
    races,
    nextRace,
    selectedSeries,
    setSelectedSeries,
    series,
    loading,
    error,
    refetch
  } = useSchedule();

  return (
    <div className="w-full">
      <Header
        title="Race Schedule"
        rightContent={
          <>
            <select
              value={selectedSeries}
              onChange={(e) => setSelectedSeries(e.target.value)}
              className="px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white backdrop-blur-sm focus:outline-none focus:border-blue-400"
            >
              {series.map((s) => (
                <option key={s.value} value={s.value} className="bg-slate-800">
                  {s.label}
                </option>
              ))}
            </select>
            
            <button
              onClick={refetch}
              disabled={loading}
              className="px-6 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded-lg font-medium transition-colors flex items-center gap-2"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
              {loading ? 'Updating...' : 'Refresh Data'}
            </button>
          </>
        }
      />

      {error && <ErrorDisplay error={error} />}

      {nextRace && <NextRaceCard nextRace={nextRace} />}

      <div className="bg-white/10 backdrop-blur-lg rounded-xl border border-white/20 overflow-hidden">
        <h2 className="text-xl font-semibold text-white p-6 border-b border-white/20">
          Full Season Schedule
        </h2>
        
        {loading ? (
          <div className="p-12 text-center text-white">
            <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-4 text-blue-400" />
            <p>Loading schedule...</p>
          </div>
        ) : (
          <div className="p-4">
            <RaceScheduleList races={races} />
          </div>
        )}
      </div>
    </div>
  );
};