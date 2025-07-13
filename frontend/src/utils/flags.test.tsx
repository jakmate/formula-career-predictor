import { describe, it, expect, vi } from 'vitest';
import { render } from '@testing-library/react';
import { CountryFlag, getFlagComponent } from './flags';

// Mock the country-flag-icons module
vi.mock('country-flag-icons/react/3x2', () => ({
  US: vi.fn(({ className }) => (
    <div data-testid="us-flag" className={className}>
      US Flag
    </div>
  )),
  GB: vi.fn(({ className }) => (
    <div data-testid="gb-flag" className={className}>
      GB Flag
    </div>
  )),
  FR: vi.fn(({ className }) => (
    <div data-testid="fr-flag" className={className}>
      FR Flag
    </div>
  )),
}));

// Mock the country codes data
vi.mock('../data/countryCodes', () => ({
  default: {
    American: 'US',
    British: 'GB',
    French: 'FR',
  },
}));

describe('CountryFlag', () => {
  it('renders flag component when nationality is valid', () => {
    const { getByTestId } = render(<CountryFlag nationality="American" />);
    const flag = getByTestId('us-flag');

    expect(flag).toBeTruthy();
    expect(flag.className).toContain('w-6 h-4');
  });

  it('applies custom className', () => {
    const customClass = 'w-8 h-6 custom-class';
    const { getByTestId } = render(
      <CountryFlag nationality="British" className={customClass} />
    );
    const flag = getByTestId('gb-flag');

    expect(flag.className).toBe(customClass);
  });

  it('applies custom className to fallback', () => {
    const customClass = 'w-8 h-6 custom-class';
    const { container } = render(
      <CountryFlag nationality={null} className={customClass} />
    );
    const fallback = container.querySelector('.bg-gray-300');

    expect(fallback?.className).toContain(customClass);
  });

  it('uses default className when none provided', () => {
    const { getByTestId } = render(<CountryFlag nationality="French" />);
    const flag = getByTestId('fr-flag');

    expect(flag.className).toBe('w-6 h-4');
  });
});

describe('getFlagComponent', () => {
  it('returns CountryFlag component with nationality prop', () => {
    const component = getFlagComponent('American');
    const { getByTestId } = render(component);
    const flag = getByTestId('us-flag');

    expect(flag).toBeTruthy();
  });

  it('returns CountryFlag component with null nationality', () => {
    const component = getFlagComponent(null);
    const { container } = render(component);
    const fallback = container.querySelector('.bg-gray-300');

    expect(fallback).toBeTruthy();
    expect(fallback?.textContent).toBe('?');
  });
});
