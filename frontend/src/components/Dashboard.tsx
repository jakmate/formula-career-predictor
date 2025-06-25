import { useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Navbar } from "./Navbar";
import { Schedule } from "./Schedule";
import { PredictionsTable } from "./table/PredictionsTable";

const Dashboard = () => {
  const navigate = useNavigate();
  const location = useLocation();
  
  // Determine active view from URL
  const activeView = location.pathname === '/schedule' ? 'schedule' : 'predictions';
  
  // Redirect to /predictions if on root path
  useEffect(() => {
    if (location.pathname === '/') {
      navigate('/predictions', { replace: true });
    }
  }, [location.pathname, navigate]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
      <Navbar activeView={activeView} />
      
      <div className="max-w-7xl mx-auto p-4">
        {activeView === 'predictions' ? <PredictionsTable /> : <Schedule />}
      </div>
    </div>
  );
};

export default Dashboard;
