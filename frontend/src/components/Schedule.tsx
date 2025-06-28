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
    refetch,
  } = useSchedule();

  return (
    <div className="w-full">
      <Header
        title="Race Schedule"
        rightContent={
          <div className="flex flex-col sm:flex-row gap-3">
            <select
              value={selectedSeries}
              onChange={(e) => setSelectedSeries(e.target.value)}
              className="px-4 py-2 bg-gray-800/60 border border-cyan-500/30 rounded-lg text-white backdrop-blur-sm focus:outline-none focus:ring-1 focus:ring-cyan-500 shadow-sm"
            >
              {series.map((s) => (
                <option key={s.value} value={s.value} className="bg-gray-900">
                  {s.label}
                </option>
              ))}
            </select>

            <button
              onClick={refetch}
              disabled={loading}
              className="px-6 py-2 bg-gradient-to-r from-cyan-600 to-purple-600 hover:from-cyan-500 hover:to-purple-500 disabled:opacity-50 text-white rounded-lg font-medium transition-all duration-200 flex items-center gap-2 shadow-lg shadow-cyan-500/20 hover:shadow-cyan-500/30"
            >
              <RefreshCw
                className={`w-4 h-4 ${loading ? "animate-spin" : ""}`}
              />
              {loading ? "Updating..." : "Refresh"}
            </button>
          </div>
        }
      />

      {error && <ErrorDisplay error={error} />}

      {nextRace && <NextRaceCard nextRace={nextRace} />}

      <div className="bg-gray-800/40 backdrop-blur-lg rounded-xl border border-cyan-500/30 overflow-hidden shadow-lg shadow-cyan-500/10">
        <h2 className="text-xl font-semibold text-white p-6 border-b border-cyan-500/20">
          Full Season Schedule
        </h2>

        {loading ? (
          <div className="p-12 text-center text-white">
            <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-4 text-cyan-400" />
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
