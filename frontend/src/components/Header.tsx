import { Calendar, RefreshCw, Target, TrendingUp, Trophy } from "lucide-react";
import type { SystemStatus } from "../types/SystemStatus";

interface HeaderProps {
    selectedModel: string;
    setSelectedModel: (model: string) => void;
    models: string[];
    loading: boolean;
    status: SystemStatus | null;
    onRefresh: () => void;
}

export const Header = ({ 
    selectedModel, 
    setSelectedModel, 
    models, 
    loading, 
    status, 
    onRefresh 
}: HeaderProps)=>(
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 mb-6 border border-white/20">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div>
            <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-2">
            <Trophy className="text-yellow-400" />
            F3 to F2 Progression Predictions
            </h1>
            <p className="text-blue-200">AI-powered analysis of Formula 3 drivers likely to advance to Formula 2</p>
        </div>
        
        <div className="flex flex-col sm:flex-row gap-3">
            <select 
            value={selectedModel} 
            onChange={(e) => setSelectedModel(e.target.value)}
            className="px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white backdrop-blur-sm focus:outline-none focus:border-blue-400"
            >
            <option value="">Select Model</option>
            {models.map(model => (
                <option key={model} value={model} className="text-gray-800">{model}</option>
            ))}
            </select>
            
            <button 
            onClick={onRefresh}
            disabled={loading}
            className="px-6 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded-lg font-medium transition-colors flex items-center gap-2"
            >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            {loading ? 'Updating...' : 'Refresh Data'}
            </button>
        </div>
        </div>
        
        {status && (
        <div className="mt-4 flex flex-wrap gap-4 text-sm text-blue-200">
            {status.last_scrape && (
            <div className="flex items-center gap-1">
                <Calendar className="w-4 h-4" />
                Last scrape: {new Date(status.last_scrape).toLocaleString()}
            </div>
            )}
            {status.last_training && (
            <div className="flex items-center gap-1">
                <TrendingUp className="w-4 h-4" />
                Last training: {new Date(status.last_training).toLocaleString()}
            </div>
            )}
            <div className="flex items-center gap-1">
            <Target className="w-4 h-4" />
            Models: {status.models_available?.length || 0}
            </div>
        </div>
        )}
    </div>
)