import { useEffect, useRef, useState } from "react";

export function Hero() {
  const sectionRef = useRef<HTMLElement>(null);
  const [scrollProgress, setScrollProgress] = useState(0);

  useEffect(() => {
    function handleScroll() {
      if (!sectionRef.current) return;
      const rect = sectionRef.current.getBoundingClientRect();
      const sectionHeight = rect.height;
      // Progress: 0 = top of section visible, 1 = section scrolled past
      const progress = Math.max(0, Math.min(1, -rect.top / (sectionHeight * 0.6)));
      setScrollProgress(progress);
    }

    window.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll();
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  // Derived animation values from scroll progress
  const translateY = scrollProgress * -60;

  return (
    <section
      ref={sectionRef}
      className="relative px-4 pb-8 pt-12 md:px-8 md:pb-16 md:pt-20"
    >
      {/* Background glow */}
      <div
        className="pointer-events-none absolute inset-0 -z-10"
        style={{ background: "var(--gradient-hero)" }}
      />

      <div className="mx-auto flex max-w-7xl flex-col items-center">
        {/* Text column — fades with scroll */}
        <div
          className="mx-auto max-w-2xl text-center transition-none"
          style={{
            opacity: Math.max(0, 1 - scrollProgress * 1.5),
            transform: `translateY(${translateY}px)`,
          }}
        >
          <p className="mb-4 text-lg font-semibold uppercase tracking-widest text-primary md:text-xl lg:text-2xl">
            AI-Powered Intelligence
          </p>
          <h1 className="font-display text-6xl font-extrabold leading-none tracking-tight sm:text-7xl md:text-8xl lg:text-[7rem]">
            VAAYU <span className="text-gradient">DARSHIYA</span>
          </h1>
          <p className="mt-6 text-xl leading-relaxed text-muted-foreground md:text-2xl lg:text-3xl">
            Real-time air quality monitoring, AI-driven pollution forecasting, and personalised
            health advisories for every Indian city.
          </p>
        </div>
      </div>
    </section>
  );
}
