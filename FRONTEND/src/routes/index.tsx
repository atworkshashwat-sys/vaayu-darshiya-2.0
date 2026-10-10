import { createFileRoute } from "@tanstack/react-router";
import Dashboard from "@/components/Dashboard";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "VAAYU DARSHIYA — AI Air Quality Forecasting" },
      {
        name: "description",
        content:
          "AI-powered air pollution forecasting with live atmospheric mapping, pollutant analytics, and best-time-to-go-outside intelligence.",
      },
      { property: "og:title", content: "VAAYU DARSHIYA — AI Air Quality Forecasting" },
      {
        property: "og:description",
        content: "Forecast air quality, visualize pollution, and find the safest hours to step outside.",
      },
    ],
    // G1: JSON-LD WebSite structured data for SEO
    scripts: [
      {
        type: "application/ld+json",
        children: JSON.stringify({
          "@context": "https://schema.org",
          "@type": "WebSite",
          name: "VAAYU DARSHIYA",
          description:
            "AI-powered air quality forecasting and live atmospheric intelligence for Indian cities.",
          url: "https://vaayu-darshiya.lovable.app",
          potentialAction: {
            "@type": "SearchAction",
            target: "https://vaayu-darshiya.lovable.app/?city={search_term_string}",
            "query-input": "required name=search_term_string",
          },
        }),
      },
    ],
  }),
  component: Dashboard,
});
