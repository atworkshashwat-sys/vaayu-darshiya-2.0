import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, type RenderOptions } from "@testing-library/react";
import type { ReactElement, ReactNode } from "react";

/**
 * Render a panel inside the React Query context the dashboard runs in.
 *
 * `Alerts` and `BestTime` are `useQuery` consumers rather than bare
 * `useState + useEffect` fetches, so a bare `render()` would throw before the
 * component draws anything. Each call builds a fresh client so one test's
 * cached response can never satisfy the next test's assertions.
 *
 * `retry: false` mirrors production behaviour for these two (a single attempt,
 * then the panel's own unavailable/stale fallback) and keeps a failing test
 * from waiting on React Query's default backoff.
 */
export function renderWithProvider(ui: ReactElement, options?: RenderOptions) {
  const client = new QueryClient({
    defaultOptions: {
      queries: { retry: false, refetchOnWindowFocus: false, gcTime: 0 },
    },
  });
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
  return render(ui, { wrapper, ...options });
}
