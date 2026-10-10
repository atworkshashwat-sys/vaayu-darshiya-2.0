import { act, renderHook } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";
import { LIVE_QUERY_KEYS, useLiveStream } from "./useLiveStream";

/**
 * The broadcast path, end to end minus the socket.
 *
 * The rule these tests hold the hook to: a card may only change because real
 * data arrived. That means (a) the payload is unwrapped from the frame the
 * server actually sends, (b) `readings` replaces the cache while `alerts` and
 * `weather` — whose frames do not match what the panels ask for — merely
 * invalidate, and (c) a dropped connection is re-established rather than
 * leaving the dashboard frozen on whatever it last saw.
 */

/** Minimal stand-in for the browser's EventSource. */
class FakeEventSource {
  static instances: FakeEventSource[] = [];
  readonly url: string;
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;
  closed = false;
  private readonly listeners = new Map<string, (event: MessageEvent<string>) => void>();

  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
  }

  addEventListener(type: string, listener: (event: MessageEvent<string>) => void): void {
    this.listeners.set(type, listener);
  }

  close(): void {
    this.closed = true;
  }

  open(): void {
    this.onopen?.();
  }

  fail(): void {
    this.onerror?.();
  }

  /** Deliver a frame in the exact shape backend `format_event()` produces. */
  emit(type: string, payload: unknown): void {
    this.raw(type, JSON.stringify({ type, server_time: "2026-10-10T19:30:00+05:30", data: payload }));
  }

  /** Deliver a body verbatim, as a truncating proxy might. */
  raw(type: string, data: string): void {
    this.listeners.get(type)?.({ data } as MessageEvent<string>);
  }
}

function setup() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, refetchOnWindowFocus: false } },
  });
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
  const rendered = renderHook(() => useLiveStream(), { wrapper });
  return { client, ...rendered };
}

beforeEach(() => {
  FakeEventSource.instances = [];
  vi.stubGlobal("EventSource", FakeEventSource);
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("useLiveStream", () => {
  it("opens exactly one connection to /api/stream", () => {
    setup();
    expect(FakeEventSource.instances).toHaveLength(1);
    expect(FakeEventSource.instances[0].url).toBe("/api/stream");
  });

  it("reports live once the server greets it", () => {
    const { result } = setup();
    act(() => FakeEventSource.instances[0].open());
    expect(result.current.status).toBe("live");
    expect(result.current.failures).toBe(0);
  });

  it("writes a readings frame straight into the stations cache", () => {
    const { client, result } = setup();
    client.setQueryData(LIVE_QUERY_KEYS.stations, [{ name: "Ashok Vihar", status: "stale" }]);

    act(() => {
      FakeEventSource.instances[0].emit("readings", {
        stations: [{ name: "Ashok Vihar", status: "live" }],
        fetched_at: "2026-10-10T19:30:00+05:30",
        offline: [],
      });
    });

    // The frame is `{type, server_time, data}`; only `data` is the payload.
    expect(client.getQueryData(LIVE_QUERY_KEYS.stations)).toEqual([
      { name: "Ashok Vihar", status: "live" },
    ]);
    expect(result.current.lastEventType).toBe("readings");
  });

  it("invalidates alerts rather than overwriting them with the wrong shape", async () => {
    const { client } = setup();
    const stored = { status: "ok", items: [{ id: "caqm-grap-3" }], sources: [] };
    client.setQueryData(LIVE_QUERY_KEYS.alerts, stored);

    await act(async () => {
      FakeEventSource.instances[0].emit("alerts", {
        items: [{ id: "live-layer-only", shape: "does-not-match" }],
        fetched_at: null,
      });
    });

    // Untouched: the panel renders an envelope this frame does not carry.
    expect(client.getQueryData(LIVE_QUERY_KEYS.alerts)).toEqual(stored);
    // But it is marked stale, so the panel re-reads its own endpoint.
    expect(client.getQueryState(LIVE_QUERY_KEYS.alerts)?.isInvalidated).toBe(true);
  });

  it("invalidates weather the same way", async () => {
    const { client } = setup();
    client.setQueryData(LIVE_QUERY_KEYS.weather, { status: "ok", slots: [] });

    await act(async () => {
      FakeEventSource.instances[0].emit("weather", { temp_c: 31, fog: false });
    });

    expect(client.getQueryData(LIVE_QUERY_KEYS.weather)).toEqual({ status: "ok", slots: [] });
    expect(client.getQueryState(LIVE_QUERY_KEYS.weather)?.isInvalidated).toBe(true);
  });

  it("stores the status frame, which carries the expired-window message", () => {
    const { client } = setup();
    act(() => {
      FakeEventSource.instances[0].emit("status", {
        active: false,
        expired: true,
        message: "Live context ended - last updated 18:00 IST",
      });
    });
    expect(client.getQueryData(LIVE_QUERY_KEYS.status)).toMatchObject({ expired: true });
  });

  it("ignores a body that is not valid JSON instead of crashing", () => {
    const { client, result } = setup();
    client.setQueryData(LIVE_QUERY_KEYS.stations, [{ name: "unchanged" }]);

    act(() => FakeEventSource.instances[0].raw("readings", "<html>proxy error</html>"));

    expect(result.current.lastEventType).toBeNull();
    expect(client.getQueryData(LIVE_QUERY_KEYS.stations)).toEqual([{ name: "unchanged" }]);
  });

  it("ignores a frame whose payload is missing", () => {
    const { client } = setup();
    client.setQueryData(LIVE_QUERY_KEYS.stations, [{ name: "unchanged" }]);

    act(() => FakeEventSource.instances[0].emit("readings", null));

    expect(client.getQueryData(LIVE_QUERY_KEYS.stations)).toEqual([{ name: "unchanged" }]);
  });

  it("reconnects with backoff rather than going silent after a drop", () => {
    vi.useFakeTimers();
    const { result } = setup();
    const first = FakeEventSource.instances[0];

    act(() => first.fail());
    expect(first.closed).toBe(true);
    expect(result.current.status).toBe("connecting");
    expect(result.current.failures).toBe(1);
    expect(FakeEventSource.instances).toHaveLength(1); // waiting on the backoff timer

    act(() => vi.advanceTimersByTime(1_000));
    expect(FakeEventSource.instances).toHaveLength(2);
    expect(FakeEventSource.instances[1].url).toBe("/api/stream");

    // A successful open resets the backoff, so a flapping server does not
    // push the retry interval out to its cap.
    act(() => FakeEventSource.instances[1].open());
    expect(result.current.status).toBe("live");
    expect(result.current.failures).toBe(0);
  });

  it("adds polling once the push channel has failed three times", async () => {
    vi.useFakeTimers();
    const { client, result } = setup();
    client.setQueryData(LIVE_QUERY_KEYS.stations, [{ name: "stale" }]);

    for (const delay of [1_000, 2_000]) {
      act(() => FakeEventSource.instances[FakeEventSource.instances.length - 1].fail());
      act(() => vi.advanceTimersByTime(delay));
    }
    act(() => FakeEventSource.instances[FakeEventSource.instances.length - 1].fail());

    expect(result.current.status).toBe("polling");
    expect(result.current.failures).toBe(3);

    // The fallback keeps the dashboard refreshing without the socket.
    await act(async () => {
      vi.advanceTimersByTime(60_000);
    });
    expect(client.getQueryState(LIVE_QUERY_KEYS.stations)?.isInvalidated).toBe(true);
  });

  it("closes the socket and cancels its timers on unmount", () => {
    vi.useFakeTimers();
    const { unmount } = setup();
    const first = FakeEventSource.instances[0];

    unmount();
    expect(first.closed).toBe(true);

    // Nothing reconnects after the component is gone.
    act(() => vi.advanceTimersByTime(60_000));
    expect(FakeEventSource.instances).toHaveLength(1);
  });
});
