import { useState, useRef, useEffect } from "react";
import { Search, MapPin, X } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

const POPULAR_CITIES = [
  "Delhi",
  "Mumbai",
  "Bengaluru",
  "Kolkata",
  "Chennai",
  "Hyderabad",
  "Ahmedabad",
  "Pune",
];

interface LocationInputProps {
  city: string;
  onSearch: (city: string) => void;
}

export function LocationInput({ city, onSearch }: LocationInputProps) {
  const [query, setQuery] = useState(city);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setQuery(city);
  }, [city]);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
        setShowSuggestions(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const filtered = POPULAR_CITIES.filter(
    (c) => c.toLowerCase().includes(query.toLowerCase()) && c !== city,
  );

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = query.trim();
    if (trimmed) {
      onSearch(trimmed);
      setShowSuggestions(false);
    }
  }

  function selectCity(c: string) {
    setQuery(c);
    onSearch(c);
    setShowSuggestions(false);
    inputRef.current?.blur();
  }

  return (
    <section id="dashboard" className="relative z-20 mx-auto -mt-6 max-w-3xl px-4 md:px-8">
      <form onSubmit={handleSubmit} ref={wrapperRef} className="relative">
        <div className="glass-strong flex items-center gap-2 rounded-2xl px-4 py-3 shadow-lg">
          <Search className="h-5 w-5 shrink-0 text-muted-foreground" />
          <Input
            ref={inputRef}
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setShowSuggestions(true);
            }}
            onFocus={() => setShowSuggestions(true)}
            placeholder="Search city or region..."
            className="border-0 bg-transparent text-base shadow-none focus-visible:ring-0 md:text-lg"
          />
          {query && (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="h-8 w-8 shrink-0"
              onClick={() => {
                setQuery("");
                inputRef.current?.focus();
              }}
            >
              <X className="h-4 w-4" />
            </Button>
          )}
          <Button type="submit" size="sm" className="shrink-0">
            Search
          </Button>
        </div>

        {/* Suggestions dropdown */}
        {showSuggestions && filtered.length > 0 && (
          <div className="glass absolute left-0 right-0 top-full z-30 mt-2 overflow-hidden rounded-xl shadow-xl">
            <p className="px-4 py-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
              Popular Cities
            </p>
            {filtered.map((c) => (
              <button
                key={c}
                type="button"
                className="flex w-full items-center gap-3 px-4 py-2.5 text-left text-sm transition-colors hover:bg-accent/10"
                onClick={() => selectCity(c)}
              >
                <MapPin className="h-4 w-4 text-primary" />
                {c}
              </button>
            ))}
          </div>
        )}
      </form>
    </section>
  );
}
