import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

/**
 * Live broadcast hook — spec §7.4.
 *
 * Opens one SSE connection to `GET /api/stream` and pushes each validated
 * refresh straight into the React Query cache, so a card updates the moment
 * the scheduler finishes a job rather than waiting for the next poll.
 *
 * Failure behaviour is deliberate:
 *   - reconnects with exponential backoff (1s → 30s), because a server restart
 *     must not leave a browser sitting on a dead socket
 *   - after 3 unsuccessful attempts it starts polling every 60 s as well, so
 *     the dashboard still refreshes if the push channel cannot be established
 *     (a proxy that blocks text/event-stream, for example)
 *   - it never shows a value it did not receive: the cache is only ever
 *     *replaced*, never merged with an optimistic guess
 *
 * SSR-safe: no socket is opened until the effect runs in a browser.
 */

export type LiveStatus = "connecting" | "live" | "polling" | "offline";

export interface LiveState {
  /** How the dashboard is currently receiving updates. */
  status: LiveStatus;
  /** ISO timestamp of the last event received over SSE, if any. */
  lastEventAt: string | null;
  /** Event type of the last event, for diagnostics. */
  lastEventType: string | null;
  /** How many consecutive connection attempts have failed. */
  failures: number;
}

const STREAM_PATH = "/api/stream";
const INITIAL_BACKOFF_MS = 1_000;
const MAX_BACKOFF_MS = 30_000;
/** Attempts before polling is added as a belt-and-braces fallback. */
const POLL_AFTER_FAILURES = 3;
const POLL_INTERVAL_MS = 60_000;

/** Query keys written by the stream — kept here so they cannot drift apart. */
export const LIVE_QUERY_KEYS = {
  stations: ["stations"],
  alerts: ["alerts"],
  weather: ["weatherWindows"],
  status: ["liveStatus"],
} as const;

export function useLiveStream(enabled = true): LiveState {
  const queryClient = useQueryClient();
  const [state, setState] = useState<LiveState>({
    status: "connecting",
    lastEventAt: null,
    lastEventType: null,
    failures: 0,
  });

  // Kept in refs so the effect below does not restart on every state change —
  // restarting it would tear down a healthy EventSource.
  const failuresRef = useRef(0);
  const backoffRef = useRef(INITIAL_BACKOFF_MS);

  useEffect(() => {
    if (!enabled || typeof window === "undefined" || !("EventSource" in window)) {
      setState((s) => ({ ...s, status: "offline" }));
      return;
    }

    let closed = false;
    let source: EventSource | null = null;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    let pollTimer: ReturnType<typeof setInterval> | null = null;

    const write = (key: readonly unknown[], payload: unknown) => {
      // A payload that is not an array where the cache holds one would break
      // every consumer, so it is ignored rather than written.
      if (payload === undefined || payload === null) return;
      queryClient.setQueryData(key, payload);
    };

    const startPolling = () => {
      if (pollTimer !== null) return;
      pollTimer = setInterval(() => {
        // Invalidate rather than fetch: each query keeps its own fallback, so
        // a backend that is down still degrades to the static snapshot.
        queryClient.invalidateQueries({ queryKey: LIVE_QUERY_KEYS.stations });
        queryClient.invalidateQueries({ queryKey: LIVE_QUERY_KEYS.alerts });
      }, POLL_INTERVAL_MS);
    };

    const scheduleRetry = () => {
      if (closed) return;
      failuresRef.current += 1;
      const delay = backoffRef.current;
      backoffRef.current = Math.min(backoffRef.current * 2, MAX_BACKOFF_MS);

      if (failuresRef.current >= POLL_AFTER_FAILURES) startPolling();
      setState((s) => ({
        ...s,
        status: failuresRef.current >= POLL_AFTER_FAILURES ? "polling" : "connecting",
        failures: failuresRef.current,
      }));

      retryTimer = setTimeout(connect, delay);
    };

    const connect = () => {
      if (closed) return;
      if (source) source.close();

      source = new EventSource(STREAM_PATH);

      source.onopen = () => {
        if (closed) return;
        failuresRef.current = 0;
        backoffRef.current = INITIAL_BACKOFF_MS;
        setState((s) => ({ ...s, status: "live", failures: 0 }));
      };

      source.onerror = () => {
        // Close first so the browser does not also run its own retry loop
        // alongside ours — two reconnect timers double the request rate.
        if (source) source.close();
        scheduleRetry();
      };

      const handle = (type: string) => (event: MessageEvent<string>) => {
        if (closed) return;
        let frame: { type?: string; server_time?: string; data?: unknown };
        try {
          frame = JSON.parse(event.data);
        } catch {
          // A frame we cannot parse is dropped; the next valid one recovers.
          return;
        }

        // Frames arrive as `{type, server_time, data}`. The payload a query
        // expects is the inner `data` — writing the wrapper itself would leave
        // every consumer reading `frame.data.stations` and finding nothing.
        const payload = frame?.data;
        if (payload === undefined || payload === null) return;

        switch (type) {
          case "readings": {
            const body = payload as { stations?: unknown };
            write(LIVE_QUERY_KEYS.stations, body.stations);
            break;
          }
          case "alerts":
          case "weather":
            // Signal, not data — the panel re-fetches from its own endpoint,
            // which is the only one that can build the envelope it renders.
            void queryClient.invalidateQueries({ queryKey: LIVE_QUERY_KEYS[type] });
            break;
          case "status":
            // The stream opens with a handshake frame of type "status"
            // carrying {kind:"connected", max_clients}. Writing that would
            // replace the REST payload this panel already holds, destroying
            // `expired`/`message` - and the header would then say "No live
            // readings yet" even after the window had ended. Only a frame
            // that actually carries status fields is allowed to replace it;
            // the handshake is acknowledged by the setState below instead.
            if (payload && typeof payload === "object" &&
                "expired" in (payload as object)) {
              write(LIVE_QUERY_KEYS.status, payload);
            }
            break;
          default:
            break;
        }

        setState((s) => ({
          ...s,
          status: "live",
          failures: 0,
          lastEventAt: new Date().toISOString(),
          lastEventType: type,
        }));
      };

      for (const type of ["readings", "alerts", "weather", "status"]) {
        source.addEventListener(type, handle(type) as EventListener);
      }
    };

    connect();

    return () => {
      closed = true;
      if (source) source.close();
      if (retryTimer) clearTimeout(retryTimer);
      if (pollTimer) clearInterval(pollTimer);
    };
  }, [enabled, queryClient]);

  return state;
}
