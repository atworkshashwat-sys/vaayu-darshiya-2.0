import { createFileRoute } from "@tanstack/react-router";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Github, Linkedin, ArrowLeft, MessageSquare } from "lucide-react";
import { Navbar } from "@/components/dashboard/Navbar";
import { Link } from "@tanstack/react-router";

export const Route = createFileRoute("/about")({
  head: () => ({
    meta: [
      { title: "About — VAAYU DARSHIYA" },
      {
        name: "description",
        content:
          "Learn about the team behind VAAYU DARSHIYA — an AI-powered air quality forecasting platform built to make air quality data actionable.",
      },
    ],
  }),
  component: AboutPage,
});

const teamMembers = [
  {
    name: "Nishant Lalwani",
    role: "Data & Feature Engineering",
    bio: "Responsible for the entire data pipeline — sourcing raw air quality and weather data, cleaning it, and transforming it into meaningful inputs for the prediction models. Designed the features that help the model understand patterns across time, seasons, and locations.",
    highlights: ["Data collection & cleaning", "Feature design", "Seasonal pattern analysis", "Training data preparation"],
    github: "https://github.com/nishant-1906",
    linkedin: "https://www.linkedin.com/in/nishant-lalwani-5a5319395/",
  },
  {
    name: "Shashwat Pandey",
    role: "Backend & Model Training",
    bio: "Built the server that powers all real-time predictions and trained the machine learning models at the core of the platform. Handled everything from model selection and tuning to creating a system that delivers forecasts instantly when a user visits the site.",
    highlights: ["Server-side architecture", "Model training & tuning", "Performance optimization", "API design"],
    github: "https://github.com/atworkshashwat-sys",
    linkedin: "https://www.linkedin.com/in/shashwat-pandey-149741384/",
  },
  {
    name: "Anvesh Kumar Bhardwaj",
    role: "Frontend & Documentation",
    bio: "Designed and built the entire user-facing experience — from interactive maps and live charts to health advisories and data exports. Focused on making complex air quality data easy to understand and visually engaging across all screen sizes.",
    highlights: ["Interactive UI development", "Map & chart visualization", "Responsive design", "Project documentation"],
    github: "https://github.com/unsigned-mog",
    linkedin: "https://www.linkedin.com/in/anvesh-kumar-bhardwaj-464879390/",
  },
];

function AboutPage() {
  return (
    <div className="min-h-screen bg-[#F8FAFC]">
      <Navbar />

      {/* Hero Header */}
      <div className="relative overflow-hidden border-b border-[#E2E8F0] bg-white">
        <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-primary/10" />
        <div className="relative mx-auto max-w-5xl px-4 py-16 sm:px-6 lg:px-8">
          <Link to="/" className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors mb-6">
            <ArrowLeft className="h-4 w-4" />
            Back to Dashboard
          </Link>
          <h1 className="font-display text-4xl font-extrabold tracking-tight text-foreground sm:text-5xl">
            About <span className="text-primary">VAAYU DARSHIYA</span>
          </h1>
          <p className="mt-4 max-w-2xl text-lg text-muted-foreground leading-relaxed">
            An AI-powered air quality intelligence platform built to make pollution data
            visible, understandable, and actionable for every Indian city.
          </p>
        </div>
      </div>

      <div className="mx-auto max-w-5xl px-4 py-16 sm:px-6 lg:px-8">
        {/* Our Story Section */}
        <section className="mb-16">
          <div className="flex items-center gap-3 mb-8">
            <div className="h-8 w-1 rounded-full bg-primary" />
            <h2 className="font-display text-3xl font-bold tracking-tight text-foreground">
              Our Story
            </h2>
          </div>
          <div className="space-y-5 text-secondary leading-[1.8] text-[15px]">
            <p>
              Vaayu Darshiya began as a college bootcamp project built by three students who wanted
              to do something about a problem they lived with every day — Delhi's air quality.
              Existing pollution dashboards often felt either too technical for everyday use or too
              vague to act on, so the team set out to build something that combined real-time
              monitoring with genuinely useful, ML-driven guidance.
            </p>
            <p>
              The project focuses on nitrogen dioxide (NO₂) and ozone (O₃), two pollutants closely
              tied to traffic emissions and sunlight-driven photochemical reactions, and which carry
              real health consequences for sensitive groups. Using historical air quality data, the
              team engineered features, trained prediction models, and built a backend capable of
              serving live forecasts alongside health-focused recommendations.
            </p>
            <p>
              What started as a class assignment grew into a full-stack application — a live map of
              monitoring stations, 24-hour pollutant trend forecasting, AI-generated insights, and
              clear health guidance for vulnerable groups. Vaayu Darshiya represents the team's
              effort to make air quality data not just visible, but genuinely actionable for the
              people who need it most.
            </p>
          </div>
        </section>

        {/* Divider */}
        <div className="flex items-center gap-4 mb-16">
          <div className="h-px flex-1 bg-[#E2E8F0]" />
          <div className="h-2 w-2 rounded-full bg-primary/40" />
          <div className="h-px flex-1 bg-[#E2E8F0]" />
        </div>

        {/* Meet the Team Section */}
        <section>
          <div className="flex items-center gap-3 mb-10">
            <div className="h-8 w-1 rounded-full bg-primary" />
            <h2 className="font-display text-3xl font-bold tracking-tight text-foreground">
              Meet the Team
            </h2>
          </div>
          <div className="grid gap-8 lg:grid-cols-3">
            {teamMembers.map((member) => (
              <Card
                key={member.name}
                className="group border-[#E2E8F0] bg-white transition-all hover:shadow-lg hover:-translate-y-1"
              >
                <CardContent className="p-7">
                  <div className="mb-5">
                    <div className="h-14 w-14 rounded-full bg-gradient-to-br from-primary/20 to-primary/5 flex items-center justify-center mb-4">
                      <span className="text-xl font-bold text-primary">
                        {member.name.split(" ").map((n) => n[0]).join("")}
                      </span>
                    </div>
                    <h3 className="font-display text-xl font-bold text-foreground">
                      {member.name}
                    </h3>
                    <p className="text-sm font-semibold text-primary mt-1">{member.role}</p>
                  </div>
                  <p className="text-sm text-secondary leading-relaxed mb-5">{member.bio}</p>
                  <div className="mb-5">
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">Key Contributions</p>
                    <div className="flex flex-wrap gap-1.5">
                      {member.highlights.map((h) => (
                        <span key={h} className="inline-block rounded-full bg-primary/8 px-2.5 py-1 text-[10px] font-medium text-primary">
                          {h}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="flex gap-2 pt-4 border-t border-[#F1F5F9]">
                    <Button
                      variant="outline"
                      size="sm"
                      asChild
                      className="gap-1.5 h-8 border-[#E2E8F0] hover:bg-[#F1F5F9]"
                    >
                      <a
                        href={member.github}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        <Github className="h-3.5 w-3.5" />
                        GitHub
                      </a>
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      asChild
                      className="gap-1.5 h-8 border-[#E2E8F0] hover:bg-[#F1F5F9]"
                    >
                      <a
                        href={member.linkedin}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        <Linkedin className="h-3.5 w-3.5" />
                        LinkedIn
                      </a>
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* Divider */}
        <div className="flex items-center gap-4 mb-16">
          <div className="h-px flex-1 bg-[#E2E8F0]" />
          <div className="h-2 w-2 rounded-full bg-primary/40" />
          <div className="h-px flex-1 bg-[#E2E8F0]" />
        </div>

        {/* Feedback & Support */}
        <section className="mb-16">
          <div className="flex items-center gap-3 mb-8">
            <div className="h-8 w-1 rounded-full bg-primary" />
            <h2 className="font-display text-3xl font-bold tracking-tight text-foreground">
              Feedback & Support
            </h2>
          </div>
          <Card className="border-[#E2E8F0] bg-white">
            <CardContent className="p-8 text-center">
              <div className="mx-auto mb-5 h-14 w-14 rounded-full bg-primary/10 flex items-center justify-center">
                <MessageSquare className="h-6 w-6 text-primary" />
              </div>
              <h3 className="font-display text-xl font-bold text-foreground mb-2">
                Found a bug or have a suggestion?
              </h3>
              <p className="text-sm text-secondary leading-relaxed mb-6 max-w-md mx-auto">
                We'd love to hear from you. Whether it's a bug report, feature request, or general feedback — your input helps us improve VAAYU DARSHIYA for everyone.
              </p>
              <Button
                size="lg"
                asChild
                className="gap-2 px-8"
              >
                <a
                  href="https://docs.google.com/forms/d/e/1FAIpQLSdvL7T847gWML14RGMx9XLt7nKKo8oEaHSiAEFnvw9YAz7HVA/viewform?usp=publish-editor"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <MessageSquare className="h-4 w-4" />
                  Submit Feedback
                </a>
              </Button>
              <p className="mt-4 text-[11px] text-muted-foreground">
                Opens a Google Form · Responses sent to atworkshashwat@gmail.com
              </p>
            </CardContent>
          </Card>
        </section>
      </div>

      {/* Footer */}
      <div className="border-t border-[#E2E8F0] bg-white">
        <div className="mx-auto max-w-5xl px-4 py-8 sm:px-6 lg:px-8">
          <p className="text-center text-sm text-muted-foreground">
            © 2026 VAAYU DARSHIYA. Built for cleaner air.
          </p>
        </div>
      </div>
    </div>
  );
}
