import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render } from '@testing-library/react';
import { CountryFlag, getFlagComponent } from './flags';
import * as Flags from 'country-flag-icons/react/3x2';

// Mock the country-flag-icons module
vi.mock('country-flag-icons/react/3x2', () => ({
  GB: vi.fn(({ className }) => (
    <div data-testid="gb-flag" className={className} />
  )),
  US: vi.fn(({ className }) => (
    <div data-testid="us-flag" className={className} />
  )),
  DE: vi.fn(({ className }) => (
    <div data-testid="de-flag" className={className} />
  )),
  FR: vi.fn(({ className }) => (
    <div data-testid="fr-flag" className={className} />
  )),
}));

// Mock the country codes data
vi.mock('../data/countryCodes', () => ({
  default: {
    British: 'GB',
    American: 'US',
    German: 'DE',
    French: 'FR',
    English: 'GB',
    Italian: 'IT',
  },
}));

describe('CountryFlag', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('fallback rendering', () => {
    it('renders fallback for null nationality', () => {
      const { container } = render(<CountryFlag nationality={null} />);
      const fallback = container.querySelector('.bg-gray-300');
      expect(fallback).toBeInTheDocument();
      expect(fallback).toHaveTextContent('?');
    });

    it('renders fallback for "Unknown" nationality', () => {
      const { container } = render(<CountryFlag nationality="Unknown" />);
      const fallback = container.querySelector('.bg-gray-300');
      expect(fallback).toBeInTheDocument();
      expect(fallback).toHaveTextContent('?');
    });

    it('renders fallback for unmapped nationality', () => {
      const { container } = render(<CountryFlag nationality="Martian" />);
      const fallback = container.querySelector('.bg-gray-300');
      expect(fallback).toBeInTheDocument();
      expect(fallback).toHaveTextContent('?');
    });

    it('applies custom className to fallback', () => {
      const { container } = render(
        <CountryFlag nationality={null} className="w-8 h-6" />
      );
      const fallback = container.querySelector('.bg-gray-300');
      expect(fallback).toHaveClass('w-8', 'h-6');
    });
  });

  describe('single flag rendering', () => {
    it('handles case-insensitive nationality matching', () => {
      const { getByTestId } = render(<CountryFlag nationality="british" />);
      expect(getByTestId('gb-flag')).toBeInTheDocument();
    });

    it('handles nationality with extra whitespace', () => {
      const { getByTestId } = render(<CountryFlag nationality="  British  " />);
      expect(getByTestId('gb-flag')).toBeInTheDocument();
    });
  });

  describe('multiple nationalities', () => {
    it('renders multiple flags for comma-separated nationalities', () => {
      const { container } = render(
        <CountryFlag nationality="British, American" />
      );
      const multiFlag = container.querySelector('.relative.inline-flex');
      expect(multiFlag).toBeInTheDocument();

      // Check that both flags are rendered
      expect(Flags.GB).toHaveBeenCalled();
      expect(Flags.US).toHaveBeenCalled();
    });

    it('renders multiple flags for dash-separated nationalities', () => {
      const { container } = render(
        <CountryFlag nationality="British-American" />
      );
      const multiFlag = container.querySelector('.relative.inline-flex');
      expect(multiFlag).toBeInTheDocument();

      expect(Flags.GB).toHaveBeenCalled();
      expect(Flags.US).toHaveBeenCalled();
    });

    it('renders multiple flags for space-separated nationalities', () => {
      const { container } = render(
        <CountryFlag nationality="British American" />
      );
      const multiFlag = container.querySelector('.relative.inline-flex');
      expect(multiFlag).toBeInTheDocument();

      expect(Flags.GB).toHaveBeenCalled();
      expect(Flags.US).toHaveBeenCalled();
    });

    it('limits to maximum 3 flags', () => {
      const { container } = render(
        <CountryFlag nationality="British, American, German, French" />
      );
      const multiFlag = container.querySelector('.relative.inline-flex');
      expect(multiFlag).toBeInTheDocument();

      // Should only render first 3 flags
      expect(Flags.GB).toHaveBeenCalled();
      expect(Flags.US).toHaveBeenCalled();
      expect(Flags.DE).toHaveBeenCalled();
      expect(Flags.FR).not.toHaveBeenCalled();
    });

    it('renders fallback when no valid flags found in multiple nationalities', () => {
      const { container } = render(
        <CountryFlag nationality="Martian, Venusian" />
      );
      const fallback = container.querySelector('.bg-gray-300');
      expect(fallback).toBeInTheDocument();
      expect(fallback).toHaveTextContent('?');
    });
  });

  describe('edge cases', () => {
    it('handles empty string nationality', () => {
      const { container } = render(<CountryFlag nationality="" />);
      const fallback = container.querySelector('.bg-gray-300');
      expect(fallback).toBeInTheDocument();
    });

    it('handles nationality with only separators', () => {
      const { container } = render(<CountryFlag nationality="- , -" />);
      const fallback = container.querySelector('.bg-gray-300');
      expect(fallback).toBeInTheDocument();
    });
  });

  describe('MultiFlag component styling', () => {
    it('applies correct positioning styles to multiple flags', () => {
      const { container } = render(
        <CountryFlag nationality="British, American" />
      );
      const flagDivs = container.querySelectorAll('.absolute');

      expect(flagDivs).toHaveLength(2);
      expect(flagDivs[0]).toHaveStyle({ left: '0%', width: '50%' });
      expect(flagDivs[1]).toHaveStyle({ left: '50%', width: '50%' });
    });

    it('applies correct positioning styles to three flags', () => {
      const { container } = render(
        <CountryFlag nationality="British, American, German" />
      );
      const flagDivs = container.querySelectorAll('.absolute');

      expect(flagDivs).toHaveLength(3);
      expect(flagDivs[0]).toHaveStyle({
        left: '0%',
        width: '33.333333333333336%',
      });
      expect(flagDivs[1]).toHaveStyle({
        left: '33.333333333333336%',
        width: '33.333333333333336%',
      });
      expect(flagDivs[2]).toHaveStyle({
        left: '66.66666666666667%',
        width: '33.333333333333336%',
      });
    });
  });
});

describe('getFlagComponent', () => {
  it('returns a CountryFlag component element', () => {
    const component = getFlagComponent('British');
    expect(component).toBeDefined();
    expect(component.type).toBe(CountryFlag);
    expect(component.props.nationality).toBe('British');
  });

  it('handles null nationality', () => {
    const component = getFlagComponent(null);
    expect(component).toBeDefined();
    expect(component.type).toBe(CountryFlag);
    expect(component.props.nationality).toBe(null);
  });
});
