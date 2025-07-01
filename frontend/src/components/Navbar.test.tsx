import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, test, vi } from 'vitest';
import { Navbar } from './Navbar';
import { MemoryRouter } from 'react-router-dom';

describe('Navbar', () => {
  beforeEach(() => {
    window.open = vi.fn();
  });

  test('opens coffee URL in new tab when clicked', async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter>
        <Navbar activeView="predictions" />
      </MemoryRouter>
    );

    const coffeeButton = screen.getByRole('button', { name: /coffee/i });
    await user.click(coffeeButton);

    expect(window.open).toHaveBeenCalledWith(
      'https://www.buymeacoffee.com/jakmate',
      '_blank'
    );
  });

  test('renders purple decorative element behind logo', () => {
    render(
      <MemoryRouter>
        <Navbar activeView="predictions" />
      </MemoryRouter>
    );

    const decorativeElement = document.querySelector(
      'div.absolute.inset-0.bg-purple-400'
    );

    expect(decorativeElement).toBeInTheDocument();
    expect(decorativeElement).toHaveClass('blur-sm');
    expect(decorativeElement).toHaveClass('opacity-40');
  });

  describe('active link styling', () => {
    test('applies active styles to predictions link when active', () => {
      render(
        <MemoryRouter>
          <Navbar activeView="predictions" />
        </MemoryRouter>
      );

      const predictionsLink = screen.getByRole('link', {
        name: /predictions/i,
      });

      expect(predictionsLink).toHaveClass('bg-gradient-to-r');
      expect(predictionsLink).toHaveClass('from-cyan-600/60');
      expect(predictionsLink).toHaveClass('to-purple-600/60');
      expect(predictionsLink).toHaveClass('text-white');
    });

    test('applies inactive styles to predictions link when not active', () => {
      render(
        <MemoryRouter>
          <Navbar activeView="schedule" />
        </MemoryRouter>
      );

      const predictionsLink = screen.getByRole('link', {
        name: /predictions/i,
      });

      expect(predictionsLink).not.toHaveClass('bg-gradient-to-r');
      expect(predictionsLink).toHaveClass('text-gray-300');
      expect(predictionsLink).toHaveClass('hover:text-white');
    });
  });
});
