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

    const coffeeLink = screen.getByRole('link', { name: /coffee/i });
    await user.click(coffeeLink);

    expect(coffeeLink).toHaveAttribute(
      'href',
      'https://www.buymeacoffee.com/jakmate'
    );
    expect(coffeeLink).toHaveAttribute('target', '_blank');
    expect(coffeeLink).toHaveAttribute('rel', 'noopener noreferrer');
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

      const promotionsLink = screen.getByRole('link', {
        name: /promotions/i,
      });

      expect(promotionsLink).toHaveClass('bg-gradient-to-r');
      expect(promotionsLink).toHaveClass('from-cyan-600');
      expect(promotionsLink).toHaveClass('to-purple-600');
      expect(promotionsLink).toHaveClass('text-white');
      expect(promotionsLink).toHaveClass('shadow-lg');
    });

    test('applies inactive styles to predictions link when not active', () => {
      render(
        <MemoryRouter>
          <Navbar activeView="schedule" />
        </MemoryRouter>
      );

      const promotionsLink = screen.getByRole('link', {
        name: /promotions/i,
      });

      expect(promotionsLink).not.toHaveClass('bg-gradient-to-r');
      expect(promotionsLink).toHaveClass('text-cyan-300');
      expect(promotionsLink).toHaveClass('hover:text-white');
    });
  });
});
