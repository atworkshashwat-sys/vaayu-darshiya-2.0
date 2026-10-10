import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Clock } from "lucide-react";
import { TIMES, formatTime } from "./data";

interface TimelineProps {
  timeIdx: number;
  setTimeIdx: (i: number) => void;
  currentMinute: number;
}

export function Timeline({ timeIdx, setTimeIdx, currentMinute }: TimelineProps) {
  return (
    <section className="mx-4 mt-8 md:mx-8">
      <Card className="glass">
        <CardHeader className="flex flex-row items-center gap-3 pb-2">
          <div className="rounded-lg bg-primary/10 p-2">
            <Clock className="h-5 w-5 text-primary" />
          </div>
          <div className="flex-1">
            <CardTitle className="text-base">Time Selector</CardTitle>
            <p className="text-xs text-muted-foreground">
              Scrub through the 24-hour forecast
            </p>
          </div>
          <div className="flex items-baseline gap-1">
            <span className="font-display text-2xl font-bold text-primary">{formatTime(timeIdx, currentMinute)}</span>
          </div>
        </CardHeader>
        <CardContent>
          <div className="px-2">
            <Slider
              value={[timeIdx]}
              onValueChange={([v]) => setTimeIdx(v)}
              min={0}
              max={23}
              step={1}
              className="w-full"
            />
            <div className="mt-2 flex justify-between text-[10px] text-muted-foreground">
              <span>12 AM</span>
              <span>6 AM</span>
              <span>12 PM</span>
              <span>6 PM</span>
              <span>11 PM</span>
            </div>
          </div>

          {/* Quick hour buttons */}
          <div className="mt-4 flex flex-wrap gap-1.5">
            {TIMES.map((t, i) => (
              <button
                key={t}
                onClick={() => setTimeIdx(i)}
                className={`rounded-md px-2 py-1 text-[10px] font-medium transition-all ${
                  i === timeIdx
                    ? "bg-primary text-primary-foreground shadow"
                    : "bg-muted text-muted-foreground hover:bg-accent/20"
                }`}
              >
                {t}
              </button>
            ))}
          </div>
        </CardContent>
      </Card>
    </section>
  );
}
