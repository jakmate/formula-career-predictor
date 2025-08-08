import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { TableHeader } from './TableHeader';
import type { SortConfig } from '../../types/Sorting';

describe('TableHeader', () => {
  const mockOnSort = vi.fn();

  beforeEach(() => {
    mockOnSort.mockClear();
  });

  it('renders children content', () => {
    const sortConfig: SortConfig = { field: 'driver', direction: 'asc' };

    render(
      <table>
        <thead>
          <tr>
            <TableHeader
              field="position"
              sortConfig={sortConfig}
              onSort={mockOnSort}
            >
              Position
            </TableHeader>
          </tr>
        </thead>
      </table>
    );

    expect(screen.getByText('Position')).toBeInTheDocument();
  });

  it('calls onSort when clicked', () => {
    const sortConfig: SortConfig = { field: 'driver', direction: 'asc' };

    render(
      <table>
        <thead>
          <tr>
            <TableHeader
              field="position"
              sortConfig={sortConfig}
              onSort={mockOnSort}
            >
              Position
            </TableHeader>
          </tr>
        </thead>
      </table>
    );

    fireEvent.click(screen.getByText('Position'));
    expect(mockOnSort).toHaveBeenCalledWith('position');
  });

  it('shows ChevronUp when active field with asc direction', () => {
    const sortConfig: SortConfig = { field: 'position', direction: 'asc' };

    render(
      <table>
        <thead>
          <tr>
            <TableHeader
              field="position"
              sortConfig={sortConfig}
              onSort={mockOnSort}
            >
              Position
            </TableHeader>
          </tr>
        </thead>
      </table>
    );

    expect(
      screen.getByRole('columnheader').querySelector('.lucide-chevron-up')
    ).toBeInTheDocument();
  });

  it('shows ChevronDown when active field with desc direction', () => {
    const sortConfig: SortConfig = { field: 'position', direction: 'desc' };

    render(
      <table>
        <thead>
          <tr>
            <TableHeader
              field="position"
              sortConfig={sortConfig}
              onSort={mockOnSort}
            >
              Position
            </TableHeader>
          </tr>
        </thead>
      </table>
    );

    expect(
      screen.getByRole('columnheader').querySelector('.lucide-chevron-down')
    ).toBeInTheDocument();
  });

  it('shows no icon when field is not active', () => {
    const sortConfig: SortConfig = { field: 'driver', direction: 'asc' };

    render(
      <table>
        <thead>
          <tr>
            <TableHeader
              field="position"
              sortConfig={sortConfig}
              onSort={mockOnSort}
            >
              Position
            </TableHeader>
          </tr>
        </thead>
      </table>
    );

    expect(screen.queryByTestId('chevron-up')).not.toBeInTheDocument();
    expect(screen.queryByTestId('chevron-down')).not.toBeInTheDocument();
  });

  it('applies correct CSS classes', () => {
    const sortConfig: SortConfig = { field: 'driver', direction: 'asc' };

    render(
      <table>
        <thead>
          <tr>
            <TableHeader
              field="position"
              sortConfig={sortConfig}
              onSort={mockOnSort}
            >
              Position
            </TableHeader>
          </tr>
        </thead>
      </table>
    );

    const th = screen.getByRole('columnheader');
    expect(th).toHaveClass(
      'p-4',
      'font-semibold',
      'cursor-pointer',
      'hover:bg-cyan-900/20',
      'transition-colors',
      'select-none'
    );
  });
});
