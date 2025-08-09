import { render } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { ProbabilityBar } from './ProbabilityBar';

describe('ProbabilityBar', () => {
  it('renders percentage text correctly', () => {
    const { getByText } = render(<ProbabilityBar percentage={75.5} />);
    expect(getByText('75.5%')).toBeInTheDocument();
  });

  it('sets correct width style', () => {
    const { container } = render(<ProbabilityBar percentage={45} />);
    const bar = container.querySelector('.h-full');
    expect(bar).toHaveStyle({ width: '45%' });
  });

  it('caps width at 100% for values over 100', () => {
    const { container } = render(<ProbabilityBar percentage={150} />);
    const bar = container.querySelector('.h-full');
    expect(bar).toHaveStyle({ width: '100%' });
  });

  it('applies green gradient for high percentages (>70)', () => {
    const { container } = render(<ProbabilityBar percentage={75} />);
    const bar = container.querySelector('.h-full');
    expect(bar).toHaveClass(
      'bg-gradient-to-r',
      'from-green-400',
      'to-cyan-400'
    );
  });

  it('applies yellow gradient for medium percentages (40-70)', () => {
    const { container } = render(<ProbabilityBar percentage={50} />);
    const bar = container.querySelector('.h-full');
    expect(bar).toHaveClass(
      'bg-gradient-to-r',
      'from-yellow-400',
      'to-amber-400'
    );
  });

  it('applies red gradient for low percentages (≤40)', () => {
    const { container } = render(<ProbabilityBar percentage={30} />);
    const bar = container.querySelector('.h-full');
    expect(bar).toHaveClass(
      'bg-gradient-to-r',
      'from-red-400',
      'to-orange-400'
    );
  });

  it('handles edge case at 40%', () => {
    const { container } = render(<ProbabilityBar percentage={40} />);
    const bar = container.querySelector('.h-full');
    expect(bar).toHaveClass(
      'bg-gradient-to-r',
      'from-red-400',
      'to-orange-400'
    );
  });

  it('handles edge case at 70%', () => {
    const { container } = render(<ProbabilityBar percentage={70} />);
    const bar = container.querySelector('.h-full');
    expect(bar).toHaveClass(
      'bg-gradient-to-r',
      'from-yellow-400',
      'to-amber-400'
    );
  });
});
