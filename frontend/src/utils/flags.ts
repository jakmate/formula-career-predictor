import React from 'react';
import * as Flags from 'country-flag-icons/react/3x2';
import NATIONALITY_TO_COUNTRY from '../data/countryCodes';

interface CountryFlagProps {
  nationality: string | null;
  className?: string;
}

const renderFallback = (className: string) => {
  return React.createElement(
    'div',
    {
      className: `${className} bg-gray-300 rounded-sm flex items-center justify-center text-xs`,
    },
    '?'
  );
};

export const CountryFlag: React.FC<CountryFlagProps> = ({
  nationality,
  className = 'w-6 h-4',
}) => {
  if (!nationality || nationality === 'Unknown') {
    return renderFallback(className);
  }

  // Get country code
  const countryCode = NATIONALITY_TO_COUNTRY[nationality];
  if (!countryCode) {
    return renderFallback(className);
  }

  // Get flag component
  const FlagComponent = Flags[countryCode as keyof typeof Flags];
  if (!FlagComponent) {
    return renderFallback(className);
  }

  return React.createElement(FlagComponent, { className });
};

export const getFlagComponent = (nationality: string | null) => {
  return React.createElement(CountryFlag, { nationality });
};
