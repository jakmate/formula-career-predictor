// Maps country names into ISO‑3166 alpha‐2
const NATIONALITY_TO_COUNTRY: Record<string, string> = {
  // Europe
  Albania: 'AL',
  Andorra: 'AD',
  Armenia: 'AM',
  Austria: 'AT',
  Belarus: 'BY',
  Belgium: 'BE',
  Bosnia: 'BA',
  Bulgaria: 'BG',
  Croatia: 'HR',
  Cyprus: 'CY',
  'Czech Republic': 'CZ',
  Estonia: 'EE',
  Finland: 'FI',
  France: 'FR',
  Georgia: 'GE',
  Germany: 'DE',
  'Great Britain': 'GB',
  Greece: 'GR',
  Hungary: 'HU',
  Iceland: 'IS',
  Ireland: 'IE',
  Italy: 'IT',
  'Kingdom Of Denmark': 'DK',
  Kosovo: 'XK',
  Latvia: 'LV',
  Liechtenstein: 'LI',
  Lithuania: 'LT',
  Luxembourg: 'LU',
  Malta: 'MT',
  Moldova: 'MD',
  Monaco: 'MC',
  Montenegro: 'ME',
  Netherlands: 'NL',
  Norway: 'NO',
  Poland: 'PL',
  Portugal: 'PT',
  Romania: 'RO',
  Russia: 'RU',
  'San Marino': 'SM',
  Serbia: 'RS',
  Slovakia: 'SK',
  Slovenia: 'SI',
  Spain: 'ES',
  Sweden: 'SE',
  Switzerland: 'CH',
  Ukraine: 'UA',
  'United Kingdom': 'GB',

  // Asia
  Bangladesh: 'BD',
  Cambodia: 'KH',
  China: 'CN',
  India: 'IN',
  Indonesia: 'ID',
  Japan: 'JP',
  Kazakhstan: 'KZ',
  Korea: 'KR',
  Kuwait: 'KW',
  Lebanon: 'LB',
  Malaysia: 'MY',
  Mongolia: 'MN',
  Nepal: 'NP',
  Pakistan: 'PK',
  "People's Republic of China": 'CN',
  Philippines: 'PH',
  Singapore: 'SG',
  'South Korea': 'KR',
  'North Korea': 'KP',
  'Sri Lanka': 'LK',
  Taiwan: 'TW',
  Thailand: 'TH',
  Uzbekistan: 'UZ',
  Vietnam: 'VN',

  // Middle East
  Afghanistan: 'AF',
  Azerbaijan: 'AZ',
  Bahrain: 'BH',
  Iran: 'IR',
  Iraq: 'IQ',
  Israel: 'IL',
  Jordan: 'JO',
  Palestine: 'PS',
  Qatar: 'QA',
  'Saudi Arabia': 'SA',
  Syria: 'SY',
  'United Arab Emirates': 'AE',
  Yemen: 'YE',

  // Oceania
  Australia: 'AU',
  'New Zealand': 'NZ',

  // North America
  Canada: 'CA',
  Mexico: 'MX',
  'United States': 'US',

  // South America
  Argentina: 'AR',
  Bolivia: 'BO',
  Brazil: 'BR',
  Chile: 'CL',
  Colombia: 'CO',
  Ecuador: 'EC',
  Guatemala: 'GT',
  Paraguay: 'PY',
  Peru: 'PE',
  Uruguay: 'UY',
  Venezuela: 'VE',
};

export default NATIONALITY_TO_COUNTRY;
