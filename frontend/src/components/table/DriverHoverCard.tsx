import React, { useState, useRef, useEffect } from 'react';
import type { Driver } from '../../types/Driver';
import { getFlagComponent } from '../../utils/flags';

interface DriverHoverCardProps {
  driver: Driver;
  children: React.ReactNode;
}

export const DriverHoverCard = ({ driver, children }: DriverHoverCardProps) => {
  const [isVisible, setIsVisible] = useState(false);
  const [position, setPosition] = useState<'top' | 'bottom'>('bottom');
  const [cardHeight, setCardHeight] = useState(500);
  const timeoutIdRef = useRef<number | null>(null);
  const cardRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLDivElement>(null);

  // Update card height when it becomes visible
  useEffect(() => {
    if (isVisible && cardRef.current) {
      setCardHeight(cardRef.current.offsetHeight);
    }
  }, [isVisible]);

  const handleMouseEnter = () => {
    // Clear any pending hide operations
    if (timeoutIdRef.current) {
      clearTimeout(timeoutIdRef.current);
      timeoutIdRef.current = null;
    }

    // Calculate position before showing
    if (triggerRef.current) {
      const rect = triggerRef.current.getBoundingClientRect();
      const spaceBelow = window.innerHeight - rect.bottom;
      const spaceAbove = rect.top;

      // Use estimated height for initial calculation
      setPosition(
        spaceBelow < cardHeight && spaceAbove > spaceBelow ? 'top' : 'bottom'
      );
    }

    // Show after delay
    timeoutIdRef.current = setTimeout(() => setIsVisible(true), 300);
  };

  const handleMouseLeave = () => {
    if (timeoutIdRef.current) {
      clearTimeout(timeoutIdRef.current);
      timeoutIdRef.current = null;
    }
    setIsVisible(false);
  };

  const driverInfo = {
    fullName: driver.driver,
    team: driver.team,
    nationality: driver.nationality,
    dateOfBirth: driver.dob,
    ...(typeof driver.age === 'number' ? { age: Math.floor(driver.age) } : {}),
    seasonWins: driver.wins,
    seasonPodiums: driver.podiums,
    photo: `https://ui-avatars.com/api/?name=${driver.driver
      .split(' ')
      .map((n) => n[0])
      .join('')}&background=1e40af&color=fff`,
  };

  return (
    <div
      className="relative inline-block"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      ref={triggerRef}
    >
      <div ref={triggerRef}>{children}</div>

      {isVisible && (
        <div
          ref={cardRef}
          className={`absolute z-50 left-0 w-80 bg-gray-800/95 backdrop-blur-lg border border-cyan-500/30 rounded-lg p-4 shadow-2xl animate-in fade-in duration-200 ${
            position === 'top' ? 'bottom-full mb-2' : 'top-full mt-2'
          }`}
        >
          <div className="flex items-start gap-4">
            <div className="relative">
              <img
                src={driverInfo.photo}
                alt={driverInfo.fullName}
                className="w-16 h-16 rounded-lg object-cover bg-slate-700 border border-cyan-500/30"
              />
              <div className="absolute -inset-1 bg-cyan-500/20 rounded-lg blur-sm z-0"></div>
            </div>

            <div className="flex-1">
              <div className="flex items-center gap-2 mb-2">
                <h3 className="text-white font-semibold text-lg">
                  {driverInfo.fullName}
                </h3>
                {getFlagComponent(driverInfo.nationality)}
              </div>

              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-white/70">Team:</span>
                  <span className="text-white">{driverInfo.team}</span>
                </div>

                <div className="flex justify-between">
                  <span className="text-white/70">DoB:</span>
                  <span className="text-white">{driverInfo.dateOfBirth}</span>
                </div>

                <div className="flex justify-between">
                  <span className="text-white/70">Age:</span>
                  <span className="text-white">{driverInfo.age}</span>
                </div>

                <div className="flex justify-between">
                  <span className="text-white/70">Experience:</span>
                  <span className="text-white">
                    {driver.experience === 0
                      ? 'Rookie'
                      : driver.experience === 1
                        ? '1 year'
                        : `${driver.experience} years`}
                  </span>
                </div>
              </div>

              <div className="mt-3 pt-2 border-t border-white/20">
                <div className="text-xs text-white/60">Current Season</div>
                <div className="flex justify-between text-sm mt-1">
                  <span className="text-white/70">Position:</span>
                  <span className="text-white">
                    {driver.position === -1 ? '-' : `#${driver.position}`}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-white/70">Points:</span>
                  <span className="text-white">{driver.points.toFixed(1)}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-white/70">Wins:</span>
                  <span className="text-white">{driverInfo.seasonWins}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-white/70">Podiums:</span>
                  <span className="text-white">{driverInfo.seasonPodiums}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
