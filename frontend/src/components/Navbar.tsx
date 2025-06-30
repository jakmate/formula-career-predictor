import { BarChart3, Calendar, Icon } from 'lucide-react';
import { motorRacingHelmet } from '@lucide/lab';
import { Link } from 'react-router-dom';

interface NavbarProps {
  activeView: string;
}

export const Navbar = ({ activeView }: NavbarProps) => {
  return (
    <nav className="bg-gray-900/80 backdrop-blur-xl border-b border-cyan-500/30 shadow-lg shadow-cyan-500/10">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex flex-col sm:flex-row items-center justify-between h-16 py-2 sm:py-0">
          <div className="flex items-center space-x-2 mb-2 sm:mb-0">
            <div className="relative">
              <Icon
                iconNode={motorRacingHelmet}
                className="w-8 h-8 text-cyan-300 z-10 relative"
              />
              <div className="absolute inset-0 bg-purple-400 rounded-full blur-sm opacity-40"></div>
            </div>
            <h1 className="text-xl font-bold text-white">
              Formula Promotion Predictions
            </h1>
          </div>

          <div className="flex space-x-1">
            <Link
              to="/predictions"
              className={`px-4 py-2 rounded-lg flex items-center space-x-2 transition-all duration-200 ${
                activeView === 'predictions'
                  ? 'bg-gradient-to-r from-cyan-600/60 to-purple-600/60 text-white shadow-lg shadow-cyan-500/20'
                  : 'text-gray-300 hover:text-white hover:bg-gray-800/50'
              }`}
            >
              <BarChart3 className="w-4 h-4" />
              <span className="hidden sm:inline">Predictions</span>
            </Link>

            <Link
              to="/schedule"
              className={`px-4 py-2 rounded-lg flex items-center space-x-2 transition-all duration-200 ${
                activeView === 'schedule'
                  ? 'bg-gradient-to-r from-cyan-600/60 to-purple-600/60 text-white shadow-lg shadow-cyan-500/20'
                  : 'text-gray-300 hover:text-white hover:bg-gray-800/50'
              }`}
            >
              <Calendar className="w-4 h-4" />
              <span className="hidden sm:inline">Schedule</span>
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
};
