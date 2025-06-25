import { BarChart3, Calendar, Trophy } from "lucide-react";
import { Link } from "react-router-dom";

interface NavbarProps {
  activeView: string;
}

export const Navbar = ({ activeView }: NavbarProps) => {
  return (
    <nav className="bg-white/10 backdrop-blur-lg border-b border-white/20">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-2">
            <Trophy className="w-8 h-8 text-blue-400" />
            <h1 className="text-xl font-bold text-white">Formula Promotion Predictions</h1>
          </div>
          
          <div className="flex space-x-1">
            <Link
              to="/predictions"
              className={`px-4 py-2 rounded-lg flex items-center space-x-2 transition-colors ${
                activeView === 'predictions'
                  ? 'bg-blue-600 text-white'
                  : 'text-white/70 hover:text-white hover:bg-white/10'
              }`}
            >
              <BarChart3 className="w-4 h-4" />
              <span>Predictions</span>
            </Link>
            
            <Link
              to="/schedule"
              className={`px-4 py-2 rounded-lg flex items-center space-x-2 transition-colors ${
                activeView === 'schedule'
                  ? 'bg-blue-600 text-white'
                  : 'text-white/70 hover:text-white hover:bg-white/10'
              }`}
            >
              <Calendar className="w-4 h-4" />
              <span>Schedule</span>
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
};
