import React from 'react';
import * as Flags from 'country-flag-icons/react/3x2';
import NATIONALITY_TO_COUNTRY from '../data/countryCodes';

interface CountryFlagProps {
  nationality: string | null;
  className?: string;
}

// Render fallback state
const renderFallback = (className: string) => {
  return React.createElement(
    'div',
    {
      className: `${className} bg-gray-300 rounded-sm flex items-center justify-center text-xs`,
    },
    '?'
  );
};

// Get valid flag components
const getValidFlags = (nationalities: string[]) => {
  const validFlags: React.ComponentType<{ className?: string }>[] = [];

  for (const nat of nationalities) {
    const countryCode = NATIONALITY_TO_COUNTRY[nat];
    if (!countryCode) continue;

    const FlagComponent = Flags[countryCode as keyof typeof Flags];
    if (FlagComponent) {
      validFlags.push(FlagComponent);
      if (validFlags.length >= 3) break; // Limit to 3 flags max
    }
  }

  return validFlags;
};

// Render multiple flags in horizontal stripes
const MultiFlag: React.FC<{
  flags: React.ComponentType<{ className?: string }>[];
  className?: string;
}> = ({ flags, className = 'w-6 h-4' }) => {
  const flagCount = flags.length;

  return React.createElement(
    'div',
    { className: `relative inline-flex ${className}` },
    flags.map((FlagComponent, index) =>
      React.createElement(
        'div',
        {
          key: index,
          className: 'absolute top-0 bottom-0',
          style: {
            left: `${(index * 100) / flagCount}%`,
            width: `${100 / flagCount}%`,
            overflow: 'hidden',
          },
        },
        React.createElement(FlagComponent, {
          className: 'w-full h-full object-cover',
        })
      )
    )
  );
};

export const CountryFlag: React.FC<CountryFlagProps> = ({
  nationality,
  className = 'w-6 h-4',
}) => {
  if (!nationality || nationality === 'Unknown') {
    return renderFallback(className);
  }

  // Split nationality string into parts
  const nationalities = nationality
    .split(/[,\-\s]+/)
    .map((n) => n.trim())
    .filter((n) => n && n !== 'Unknown')
    .map((n) => n.charAt(0).toUpperCase() + n.slice(1).toLowerCase());

  // Try to get valid flags from the parts
  let validFlags = getValidFlags(nationalities);

  // If no flags found, try the entire string
  if (validFlags.length === 0) {
    const countryCode = NATIONALITY_TO_COUNTRY[nationality];
    if (countryCode) {
      const FlagComponent = Flags[countryCode as keyof typeof Flags];
      if (FlagComponent) {
        validFlags = [FlagComponent];
      }
    }
  }

  // Render based on number of valid flags
  if (validFlags.length === 0) {
    return renderFallback(className);
  } else if (validFlags.length === 1) {
    return React.createElement(validFlags[0], { className });
  } else {
    return React.createElement(MultiFlag, { flags: validFlags, className });
  }
};

export const getFlagComponent = (nationality: string | null) => {
  return React.createElement(CountryFlag, { nationality });
};
