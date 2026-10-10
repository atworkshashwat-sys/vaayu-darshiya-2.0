// src/test-setup.ts — Global test environment stubs
import "@testing-library/jest-dom";

// Stub window.matchMedia (not in jsdom by default)
Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  }),
});

// Stub ResizeObserver (used by recharts / framer-motion)
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
window.ResizeObserver = ResizeObserverStub;

// Stub IntersectionObserver (used by framer-motion viewport detection)
class IntersectionObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
(window as unknown as { IntersectionObserver: unknown }).IntersectionObserver =
  IntersectionObserverStub;
