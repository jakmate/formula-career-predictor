import { render } from '@testing-library/react';
import App from './App';
import { test } from 'vitest';

test('renders app', () => {
  render(<App />);
});
