// AQI helper utilities — CPCB National Air Quality Index (India)
// Scale: Good 0–50, Satisfactory 51–100, Moderate 101–200,
//        Poor 201–300, Very Poor 301–400, Severe 401–500

export type AqiCategory = {
  label: string;
  range: string;
  color: string;
  textColor: string;
  advice: string;
};

export const AQI_BREAKPOINTS = [
  { min: 0, max: 50, label: "Good", color: "#00c853", textColor: "#003d00" },
  { min: 51, max: 100, label: "Satisfactory", color: "#ffc107", textColor: "#5d4000" },
  { min: 101, max: 200, label: "Moderate", color: "#ff9800", textColor: "#4e2600" },
  { min: 201, max: 300, label: "Poor", color: "#f44336", textColor: "#4a0000" },
  { min: 301, max: 400, label: "Very Poor", color: "#9c27b0", textColor: "#2a0033" },
  { min: 401, max: 500, label: "Severe", color: "#7e0023", textColor: "#3b0013" },
] as const;

export function getAqiCategory(aqi: number): AqiCategory {
  if (aqi <= 50) return { ...AQI_BREAKPOINTS[0], range: "0–50" };
  if (aqi <= 100) return { ...AQI_BREAKPOINTS[1], range: "51–100" };
  if (aqi <= 200) return { ...AQI_BREAKPOINTS[2], range: "101–200" };
  if (aqi <= 300) return { ...AQI_BREAKPOINTS[3], range: "201–300" };
  if (aqi <= 400) return { ...AQI_BREAKPOINTS[4], range: "301–400" };
  return { ...AQI_BREAKPOINTS[5], range: "401–500" };
}

export function getAqiColor(aqi: number): string {
  return getAqiCategory(aqi).color;
}

export function getAqiLabel(aqi: number): string {
  return getAqiCategory(aqi).label;
}

export function getAqiTextColor(aqi: number): string {
  return getAqiCategory(aqi).textColor;
}

export function getAqiEmoji(aqi: number): string {
  if (aqi <= 50) return "\u{1F7E2}";
  if (aqi <= 100) return "\u{1F7E1}";
  if (aqi <= 200) return "\u{1F7E0}";
  if (aqi <= 300) return "\u{1F534}";
  if (aqi <= 400) return "\u{1F7E3}";
  return "\u{1F7E3}";
}

// Health advice per CPCB category
export function getHealthAdvice(aqi: number): string {
  if (aqi <= 50)
    return "Air quality is satisfactory. Enjoy outdoor activities.";
  if (aqi <= 100)
    return "Acceptable air quality. Unusually sensitive people should consider reducing prolonged outdoor exertion.";
  if (aqi <= 200)
    return "Moderate risk. Sensitive groups (children, elderly, asthmatics) should reduce prolonged outdoor exertion.";
  if (aqi <= 300)
    return "Poor air quality. Everyone should avoid prolonged outdoor exertion. Sensitive groups should remain indoors.";
  if (aqi <= 400)
    return "Very poor air quality. Avoid all outdoor activities. Keep windows closed. Use air purifiers indoors.";
  return "Severe air quality. Emergency conditions. Avoid all outdoor exertion. Wear N95 masks if going outside.";
}
