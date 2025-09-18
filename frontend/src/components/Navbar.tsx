import { Calendar, Icon, Coffee, TrendingUp } from 'lucide-react';
import { motorRacingHelmet } from '@lucide/lab';
import { Link } from 'react-router-dom';

interface NavbarProps {
  activeView: 'predictions' | 'schedule';
}

const BuyMeCoffee = () => {
  return (
    <a
      href="https://www.buymeacoffee.com/jakmate"
      target="_blank"
      rel="noopener noreferrer"
      className="px-6 py-2 rounded-lg font-medium transition-all duration-200 flex items-center gap-2 text-cyan-300 hover:text-white hover:bg-gray-800/50"
    >
      <Coffee className="w-4 h-4" />
      <span>Coffee</span>
    </a>
  );
};

export const Navbar = ({ activeView }: NavbarProps) => {
  const isActive = (view: string) => activeView === view;

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
              Formula Predictions
            </h1>
          </div>

          <div className="flex space-x-1">
            <Link
              to="/predictions"
              className={`px-6 py-2 rounded-lg font-medium transition-all duration-200 flex items-center gap-2 ${
                isActive('predictions')
                  ? 'bg-gradient-to-r from-cyan-600 to-purple-600 text-white shadow-lg shadow-cyan-500/20'
                  : 'text-cyan-300 hover:text-white hover:bg-gray-800/50'
              }`}
            >
              <TrendingUp className="w-4 h-4" />
              Promotions
            </Link>

            <Link
              to="/schedule"
              className={`px-6 py-2 rounded-lg font-medium transition-all duration-200 flex items-center gap-2 ${
                isActive('schedule')
                  ? 'bg-gradient-to-r from-cyan-600 to-purple-600 text-white shadow-lg shadow-cyan-500/20'
                  : 'text-cyan-300 hover:text-white hover:bg-gray-800/50'
              }`}
            >
              <Calendar className="w-4 h-4" />
              Schedule
            </Link>

            <BuyMeCoffee></BuyMeCoffee>
          </div>
        </div>
      </div>
    </nav>
  );
};
