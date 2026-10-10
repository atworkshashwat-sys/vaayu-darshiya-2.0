"""
Validation for live web readings — reject, never guess (spec §3).

Everything TinyFish returns is untrusted text. This module is the only place a
fetched value is turned into a number, and it refuses to invent one:

  * range checks      NO2 0-500 ppb, O3 0-300 ppb, PM2.5 0-1000 µg/m³, AQI 0-500
  * unit handling     CPCB/DPCC publish gas concentrations in µg/m³; the app
                      works in ppb, so those are converted with the documented
                      molar factor below. An unknown unit is REJECTED, never
                      assumed to be ppb.
  * timestamps        converted to IST at ingestion; a reading older than
                      STALE_AFTER_MINUTES is marked "stale" instead of being
                      shown as current
  * disagreement      two sources more than 30% apart → keep the official CPCB
                      value and flag the row as "conflict"
  * never overwrite   live readings live in their own fields; they can never
                      replace an ML forecast value

Unit conversion factor
----------------------
CPCB and DPCC publish NO₂ and O₃ as mass concentrations in µg/m³. The model,
the map and the tables all work in ppb (parts per billion, volume mixing
ratio). The standard conversion at 25 °C and 1 atm uses the molar volume
V = 24.45 L/mol:

    ppb = (µg/m³) / (M / V) = (µg/m³) × V / M

    NO₂  M = 46.01 g/mol  →  1 ppb = 1.882 µg/m³  →  µg/m³ ÷ 1.882
    O₃   M = 48.00 g/mol  →  1 ppb = 1.963 µg/m³  →  µg/m³ ÷ 1.963

PM2.5 has no gas-phase conversion (it is already a mass concentration and the
CPCB AQI sub-index is defined on it in µg/m³) and is stored as-is. AQI is a
unitless index and is never converted.

Recomputing the factor at the reading's own temperature is possible but would
make a single value jump around between refreshes for no visible reason; 25 °C
is the convention CPCB itself uses for its published conversions, so it is
fixed here and documented rather than silently varied.
"""

import re
from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))

# Molar volume of ideal gas at 25 °C, 1 atm (L/mol) — the CPCB convention.
MOLAR_VOLUME = 24.45

# µg/m³ → ppb divisors (M / V). See the module docstring for the derivation.
MOLAR_MASS = {"NO2": 46.01, "O3": 48.00}
PPB_PER_UG = {pollutant: mass / MOLAR_VOLUME for pollutant, mass in MOLAR_MASS.items()}

# Accepted unit spellings → canonical form.
UNIT_ALIASES = {
    "ppb": "ppb", "parts per billion": "ppb",
    "ug/m3": "ug/m3", "µg/m3": "ug/m3", "μg/m3": "ug/m3",
    "ug/m³": "ug/m3", "µg/m³": "ug/m3", "μg/m³": "ug/m3",
    "mcg/m3": "ug/m3", "micrograms/m3": "ug/m3",
    "µg/m^3": "ug/m3", "ugm-3": "ug/m3",
    "index": "index", "aqi": "index", "cpcb aqi": "index",
    "": "index",   # AQI is the only unitless field; callers must still match
}

# Per-pollutant bounds (spec §3) in the canonical unit.
LIMITS = {
    "NO2":  {"unit": "ppb",    "min": 0.0,   "max": 500.0},
    "O3":   {"unit": "ppb",    "min": 0.0,   "max": 300.0},
    "PM25": {"unit": "ug/m3",  "min": 0.0,   "max": 1000.0},
    "AQI":  {"unit": "index",  "min": 0.0,   "max": 500.0},
}

# A reading older than this is served, but labelled "stale" (spec §3).
STALE_AFTER_MINUTES = 180
# Two sources this far apart (relative) are in conflict (spec §3).
CONFLICT_RELATIVE = 0.30


class Rejected(ValueError):
    """A fetched value that could not be trusted. Carries the reason."""


def canonical_unit(raw):
    """Normalise a unit string, or None when it is not one we understand."""
    if raw is None:
        return None
    return UNIT_ALIASES.get(str(raw).strip().lower())


def to_ppb(pollutant, value, unit):
    """Convert a gas concentration to ppb. Returns (value, unit).

    Raises Rejected for an unknown unit or an unknown pollutant — the caller
    must then keep its previous value rather than guess.
    """
    canonical = canonical_unit(unit)
    if pollutant not in PPB_PER_UG:
        raise Rejected(f"no conversion defined for {pollutant}")
    if canonical == "ppb":
        return float(value), "ppb"
    if canonical == "ug/m3":
        return round(float(value) / PPB_PER_UG[pollutant], 2), "ppb"
    raise Rejected(f"unrecognised unit {unit!r} for {pollutant}")


def parse_number(raw):
    """Pull the first plausible number out of fetched text.

    Handles "34", "34.8", "<1", ">900", "1,204" and unicode minus. Returns None
    when there is no number at all — never a zero, never a guess.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    text = str(raw).strip()
    if not text:
        return None
    text = text.replace("\u2212", "-").replace(",", "")
    # "<1" and ">900" are censoring marks; the bound is the honest reading.
    text = text.lstrip("<>≈~ ")
    for token in text.replace("/", " ").replace(";", " ").split():
        cleaned = token.strip(".,;:%()[]{}")
        try:
            return float(cleaned)
        except ValueError:
            continue
    return None


def validate(pollutant, value, unit, observed_at=None, source="", now=None):
    """Validate one fetched reading into the canonical reading schema.

    Returns a dict ready for live_store.save_readings(), or raises Rejected.
    Nothing is coerced: an unparseable value, an unknown unit, an out-of-range
    number or an unparseable timestamp all raise, and the caller keeps its
    previous (possibly stale) value.
    """
    limit = LIMITS.get(pollutant)
    if limit is None:
        raise Rejected(f"unknown pollutant {pollutant!r}")

    if value is None:
        raise Rejected("no numeric value could be read")

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise Rejected(f"value {value!r} is not numeric")
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        raise Rejected("value is not finite")

    # Unit conversion (NO2/O3 arrive as µg/m³ from CPCB/DPCC).
    if limit["unit"] == "ppb":
        numeric, unit_out = to_ppb(pollutant, numeric, unit)
    else:
        canonical = canonical_unit(unit)
        if canonical is None:
            raise Rejected(f"unrecognised unit {unit!r} for {pollutant}")
        if canonical != limit["unit"]:
            raise Rejected(f"unit {unit!r} is not {limit['unit']} for {pollutant}")
        unit_out = canonical
        numeric = round(float(numeric), 1)

    if not (limit["min"] <= numeric <= limit["max"]):
        raise Rejected(
            f"{pollutant}={numeric} {unit_out} outside "
            f"{limit['min']}-{limit['max']} {limit['unit']}")

    observed = observed_at if isinstance(observed_at, datetime) else None
    if observed is None and isinstance(observed_at, str):
        observed = _parse_timestamp(observed_at)
    if observed is None:
        raise Rejected("observed_at is missing or unparseable")
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=IST)
    observed = observed.astimezone(IST)

    return {
        "pollutant": pollutant,
        "value": numeric,
        "unit": unit_out,
        "observed_at": observed.isoformat(timespec="seconds"),
        "source": str(source or "unknown"),
        "fetched_at": (now or datetime.now(IST)).isoformat(timespec="seconds"),
    }


def _parse_timestamp(raw):
    """Best-effort IST timestamp from source text. None rather than a guess.

    Handles the formats the live sources actually emit, including the DPCC
    automation output ("Saturday, October 10, 2026 18:25:00 IST") — a leading
    weekday and a trailing zone abbreviation are stripped before matching,
    because neither is a real timezone offset.
    """
    text = str(raw or "").strip()
    if not text:
        return None
    # "Saturday, " / "Sat " prefix and " IST" / " UTC" suffix carry no offset
    # we can trust, so they are removed rather than mis-parsed as local time.
    text = re.sub(
        r"^(?:mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)[a-z]*,?\s+",
        "", text, flags=re.I)
    text = re.sub(
        r"\s+(?:ist|utc|gmt|pst|pdt|cst|cdt|bst|jst|cet)$",
        "", text, flags=re.I).strip()
    # CPCB's city report writes "Oct 10, 2026 @ 04 PM" — the "@" is decoration.
    text = text.replace("@", " ").strip()

    cleaned = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        parsed = None
    if parsed is None:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d %b %Y %H:%M",
                    "%d %B %Y, %H:%M", "%d %B %Y %H:%M:%S", "%d %B %Y, %H:%M:%S",
                    "%d %b %Y, %H:%M", "%B %d, %Y %H:%M:%S", "%B %d, %Y %H:%M",
                    "%b %d, %Y %I %p", "%b %d, %Y %I:%M %p", "%b %d, %Y %H:%M",
                    "%d %b %Y", "%Y-%m-%d"):
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
    return parsed


def age_minutes(observed_at, now=None):
    """Whole minutes since the reading was taken; None when unknown."""
    observed = observed_at if isinstance(observed_at, datetime) else _parse_timestamp(observed_at)
    if observed is None:
        return None
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=IST)
    reference = now or datetime.now(IST)
    return max(0, int((reference - observed.astimezone(IST)).total_seconds() // 60))


def freshness(observed_at, now=None):
    """'live' | 'stale' | 'offline' from the reading's age (spec §3)."""
    age = age_minutes(observed_at, now)
    if age is None:
        return "offline"
    return "live" if age <= STALE_AFTER_MINUTES else "stale"


def resolve_conflict(candidate, existing, tolerance=CONFLICT_RELATIVE):
    """Decide between two sources for the same station+pollutant.

    `candidate` and `existing` are validated reading dicts. Returns
    (reading, conflicted) where `reading` is the one to keep.

    Rule (spec §3): when two sources disagree by more than `tolerance`
    relative, keep the OFFICIAL CPCB value and flag the row as a conflict.
    With one source, or two that agree, there is no conflict.
    """
    if not existing:
        return candidate, False

    base = abs(candidate["value"])
    other = abs(existing["value"])
    reference = max(abs(base), abs(other))
    if reference == 0:
        agrees = base == other
    else:
        agrees = abs(base - other) / reference <= tolerance

    if agrees:
        # Prefer the newer reading when they agree.
        if candidate.get("observed_at", "") >= existing.get("observed_at", ""):
            return candidate, False
        return existing, False

    official = _official_rank(candidate.get("source")), _official_rank(existing.get("source"))
    keep = candidate if official[0] <= official[1] else existing
    keep = dict(keep)
    keep["conflict"] = 1
    return keep, True


def _official_rank(source):
    """Lower is more official. CPCB > DPCC > CAQM/IMD/SAFAR > press > other."""
    name = str(source or "").lower()
    if "cpcb" in name:
        return 0
    if "dpcc" in name:
        return 1
    if any(tag in name for tag in ("caqm", "imd", "safar", "iitm", "pib")):
        return 2
    if any(tag in name for tag in ("openaq", "waqi")):
        return 3
    return 9
