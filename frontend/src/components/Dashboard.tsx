import { useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Navbar } from './Navbar';
import { Schedule } from './Schedule';
import { PredictionsTable } from './table/PredictionsTable';

const Dashboard = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const activeView =
    location.pathname === '/schedule' ? 'schedule' : 'predictions';

  useEffect(() => {
    if (location.pathname === '/') {
      navigate('/predictions', { replace: true });
    }
  }, [location.pathname, navigate]);

  return (
    <div className="min-h-screen bg-gray-900 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-gray-800 via-gray-900 to-black">
      <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI1MCIgaGVpZ2h0PSI1MCI+PHBhdGggZD0iTTAgMGg1MHY1MEgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGQ9Ik0wIDBoNXY1TDAgNXpNNTAgNTB2LTVsLTUgNXpNNTAgMHY1bDUtNXpNMCA1MGw1LTV2NWgtNXoiIHN0cm9rZT0iIzAwZmZmZiIgc3Ryb2tlLW9wYWNpdHk9IjAuMSIgc3Ryb2tlLXdpZHRoPSIxIi8+PC9zdmc+')] opacity-5"></div>

      <Navbar activeView={activeView} />

      <div className="max-w-7xl mx-auto p-4 relative z-10">
        {activeView === 'predictions' ? <PredictionsTable /> : <Schedule />}
      </div>
    </div>
  );
};

export default Dashboard;
