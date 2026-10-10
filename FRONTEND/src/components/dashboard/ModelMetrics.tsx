import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  BarChart3, TrendingUp, Target, Gauge,
  CheckCircle2, Layers, Zap, Database, Cpu, Award
} from "lucide-react";
import { fetchMetrics, fetchModels, postModelOverride, ModelMetricsData, ModelsData, ModelBlock, ModelName } from "./data";

const HORIZON_ROWS = [0, 5, 11, 23]; // t+1, t+6, t+12, t+24 (indices into horizon arrays)

function fmt(v: number | null | undefined, digits = 3): string {
  return v == null || Number.isNaN(v) ? "—" : v.toFixed(digits);
}

export function ModelMetrics() {
  const [data, setData] = useState<ModelMetricsData | null>(null);
  const [models, setModels] = useState<ModelsData | null>(null);
  const [metricTab, setMetricTab] = useState("overview");

  useEffect(() => {
    fetchMetrics().then(setData);
    fetchModels().then(setModels);
  }, []);

  const handleOverride = async (model: ModelName | "auto") => {
    const res = await postModelOverride(model);
    if (res.ok) fetchModels().then(setModels);
  };

  if (!data) {
    return (
      <section className="mx-4 mt-8 md:mx-8">
        <div className="mb-6 flex items-center gap-3">
          <div className="rounded-xl bg-secondary/10 p-2.5">
            <BarChart3 className="h-6 w-6 text-secondary" />
          </div>
          <div>
            <h2 className="font-display text-2xl font-bold">Model Performance</h2>
            <p className="text-sm text-muted-foreground">Loading model metrics...</p>
          </div>
        </div>
      </section>
    );
  }

  const m = data.metrics;
  const fi = data.feature_importance;
  const hp = data.hyperparameters;
  const t = data.training;
  const vc = m.validation_comparison;

  // Accuracy of the same predictions against data/vaayu_final_v3.csv,
  // written by scripts/evaluate_accuracy.py.
  const dbVal = m.database_validation;
  const dbRows = dbVal
    ? (["NO2", "O3", "HCHO"] as const).flatMap((tgt) => {
        const blk = dbVal.targets[tgt];
        if (!blk) return [];
        return [
          { id: `${tgt}-test`, target: tgt, basis: "Model's own labels", score: blk.model_test.all, own: true },
          { id: `${tgt}-db`, target: tgt, basis: "Database observed", score: blk.database.all, own: false },
        ];
      })
    : [];

  const keyMetrics = [
    {
      label: "NO2 R2 (t+1h)",
      value: m.t1h.NO2.r2.toFixed(4),
      icon: TrendingUp,
      color: "text-emerald-400",
      desc: `MAE ${m.t1h.NO2.mae} ppb  RMSE ${m.t1h.NO2.rmse} ppb`,
    },
    {
      label: "O3 R2 (t+1h)",
      value: m.t1h.O3.r2.toFixed(4),
      icon: TrendingUp,
      color: "text-sky-400",
      desc: `MAE ${m.t1h.O3.mae} ppb  RMSE ${m.t1h.O3.rmse} ppb`,
    },
    {
      label: "NO2 R2 (t+6h)",
      value: m.t6h.NO2.r2.toFixed(4),
      icon: Target,
      color: "text-amber-400",
      desc: `MAE ${m.t6h.NO2.mae} ppb  RMSE ${m.t6h.NO2.rmse} ppb`,
    },
    {
      label: "O3 R2 (t+6h)",
      value: m.t6h.O3.r2.toFixed(4),
      icon: Target,
      color: "text-amber-400",
      desc: `MAE ${m.t6h.O3.mae} ppb  RMSE ${m.t6h.O3.rmse} ppb`,
    },
    {
      label: "NO2 R2 (t+12h)",
      value: m.t12h.NO2.r2.toFixed(4),
      icon: Gauge,
      color: "text-orange-400",
      desc: `MAE ${m.t12h.NO2.mae} ppb  RMSE ${m.t12h.NO2.rmse} ppb`,
    },
    {
      label: "O3 R2 (t+12h)",
      value: m.t12h.O3.r2.toFixed(4),
      icon: Gauge,
      color: "text-orange-400",
      desc: `MAE ${m.t12h.O3.mae} ppb  RMSE ${m.t12h.O3.rmse} ppb`,
    },
    ...(m.t1h.HCHO && m.t12h.HCHO
      ? [
          {
            label: "HCHO R2 (t+1h)",
            value: m.t1h.HCHO.r2.toFixed(4),
            icon: TrendingUp,
            color: "text-violet-400",
            desc: `MAE ${m.t1h.HCHO.mae} µmol/m²  RMSE ${m.t1h.HCHO.rmse} µmol/m²`,
          },
          {
            label: "HCHO R2 (t+12h)",
            value: m.t12h.HCHO.r2.toFixed(4),
            icon: Gauge,
            color: "text-violet-400",
            desc: `MAE ${m.t12h.HCHO.mae} µmol/m²  RMSE ${m.t12h.HCHO.rmse} µmol/m²`,
          },
        ]
      : []),
  ];

  const bestHorizons = [0, 1, 5, 11];

  return (
    <section className="mx-4 mt-8 md:mx-8">
      {/* Header */}
      <div className="mb-6 flex items-center gap-3">
        <div className="rounded-xl bg-secondary/10 p-2.5">
          <BarChart3 className="h-6 w-6 text-secondary" />
        </div>
        <div>
          <h2 className="font-display text-2xl font-bold">Model Performance</h2>
          <p className="text-sm text-muted-foreground">
            {data.model} — XGBoost per-horizon models for NO2, O3 and HCHO prediction
            <span className="mt-1 block text-[10px] text-muted-foreground">
              {t.sites} Delhi sites  {t.features} features (NO2/O3) · {t.features_hcho ?? "100"} (HCHO)  {t.horizons} forecast horizons  Train {t.train_years}  Val {t.val_years}  Test {t.test_years}
            </span>
          </p>
        </div>
      </div>

      {/* 3 Tabs */}
      <Tabs value={metricTab} onValueChange={setMetricTab} className="space-y-6">
        <TabsList className="glass">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="models">Models</TabsTrigger>
          <TabsTrigger value="validation">Validation</TabsTrigger>
          <TabsTrigger value="methodology">Methodology</TabsTrigger>
        </TabsList>

        {/* ═══════════════ OVERVIEW TAB ═══════════════ */}
        <TabsContent value="overview" className="space-y-6">
          {/* Key Metric Cards */}
          <div className="grid gap-4 md:grid-cols-3 lg:grid-cols-4">
            {keyMetrics.map((km) => (
              <Card key={km.label} className="glass overflow-hidden transition-all hover:shadow-lg">
                <CardHeader className="flex flex-row items-center justify-between pb-1">
                  <CardTitle className="text-xs font-medium text-muted-foreground">{km.label}</CardTitle>
                  <div className="rounded-lg bg-secondary/10 p-1.5">
                    <km.icon className={`h-3.5 w-3.5 ${km.color}`} />
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="flex items-baseline gap-1">
                    <span className={`text-2xl font-bold ${km.color}`}>{km.value}</span>
                  </div>
                  <p className="mt-1.5 text-[10px] leading-relaxed text-muted-foreground">{km.desc}</p>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Best Horizons Performance Table */}
          <Card className="glass">
            <CardHeader>
              <CardTitle className="text-base">Per-Horizon Performance (Test Set)</CardTitle>
              <p className="text-xs text-secondary">
                R2, MAE, and RMSE for each forecast horizon from t+1h to t+24h
              </p>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="px-2 py-1.5 text-left font-medium text-muted-foreground">Horizon</th>
                      <th className="px-2 py-1.5 text-right font-medium text-emerald-400">NO2 R2</th>
                      <th className="px-2 py-1.5 text-right font-medium text-emerald-400">NO2 MAE</th>
                      <th className="px-2 py-1.5 text-right font-medium text-emerald-400">NO2 RMSE</th>
                      <th className="px-2 py-1.5 text-right font-medium text-sky-400">O3 R2</th>
                      <th className="px-2 py-1.5 text-right font-medium text-sky-400">O3 MAE</th>
                      <th className="px-2 py-1.5 text-right font-medium text-sky-400">O3 RMSE</th>
                      <th className="px-2 py-1.5 text-right font-medium text-violet-400">HCHO R2</th>
                      <th className="px-2 py-1.5 text-right font-medium text-violet-400">HCHO MAE</th>
                      <th className="px-2 py-1.5 text-right font-medium text-violet-400">HCHO RMSE</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bestHorizons.map((hIdx) => {
                      const no2Row = m.all_horizons.NO2[hIdx];
                      const o3Row = m.all_horizons.O3[hIdx];
                      const hchoRow = m.all_horizons.HCHO?.[hIdx];
                      if (!no2Row) return null;
                      const h = no2Row.horizon + 1;
                      return (
                        <tr key={no2Row.horizon} className="border-b border-border/50 hover:bg-muted/30">
                          <td className="px-2 py-1.5 font-medium">t+{h}h</td>
                          <td className="px-2 py-1.5 text-right font-mono">{no2Row.r2.toFixed(3)}</td>
                          <td className="px-2 py-1.5 text-right font-mono">{no2Row.mae}</td>
                          <td className="px-2 py-1.5 text-right font-mono">{no2Row.rmse}</td>
                          <td className="px-2 py-1.5 text-right font-mono">{o3Row?.r2.toFixed(3)}</td>
                          <td className="px-2 py-1.5 text-right font-mono">{o3Row?.mae}</td>
                          <td className="px-2 py-1.5 text-right font-mono">{o3Row?.rmse}</td>
                          <td className="px-2 py-1.5 text-right font-mono">{hchoRow ? hchoRow.r2.toFixed(3) : "—"}</td>
                          <td className="px-2 py-1.5 text-right font-mono">{hchoRow?.mae ?? "—"}</td>
                          <td className="px-2 py-1.5 text-right font-mono">{hchoRow?.rmse ?? "—"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* ═══════════════ MODELS TAB (multi-model ensemble) ═══════════════ */}
        <TabsContent value="models" className="space-y-6">
          {!models || models.models.length === 0 ? (
            <Card className="glass">
              <CardContent className="py-8 text-center text-sm text-muted-foreground">
                Ensemble unavailable — the backend did not return any models.
              </CardContent>
            </Card>
          ) : (
            <>
              {/* Active model + data source + override control */}
              <Card className="glass">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Cpu className="h-4 w-4 text-secondary" />
                    Active Model
                    <Badge className="ml-auto bg-emerald-500/15 text-emerald-400">
                      {models.active_model}
                    </Badge>
                    <Badge variant="outline" className="text-[10px]">
                      Data: {models.data_source}
                    </Badge>
                  </CardTitle>
                  <p className="text-xs text-secondary">
                    {models.active_reason} · {models.data_source_detail}
                  </p>
                </CardHeader>
                <CardContent className="flex flex-wrap items-center gap-2">
                  <span className="text-xs text-muted-foreground">Manual override:</span>
                  <select
                    value={models.override ?? "auto"}
                    onChange={(e) => handleOverride(e.target.value as ModelName | "auto")}
                    className="rounded-md border border-border/60 bg-background px-2 py-1 text-xs"
                  >
                    <option value="auto">Auto (best rolling RMSE)</option>
                    {models.models
                      .filter((mm) => mm.available)
                      .map((mm) => (
                        <option key={mm.name} value={mm.name}>{mm.name}</option>
                      ))}
                  </select>
                  <span className="text-[10px] text-muted-foreground">
                    Rolling window: {models.rolling_window_days} days · Fallback: &gt;
                    {Math.round(models.fallback_rules.missing_features_threshold * 100)}% missing → RF,
                    &lt;{models.fallback_rules.min_features_ridge} features → Ridge
                  </span>
                </CardContent>
              </Card>

              {/* One card per model */}
              <div className="grid gap-4 lg:grid-cols-2">
                {models.models.map((mb) => (
                  <ModelCard key={mb.name} block={mb} />
                ))}
              </div>
            </>
          )}
        </TabsContent>

        {/* ═══════════════ VALIDATION TAB ═══════════════ */}
        <TabsContent value="validation" className="space-y-6">
          {vc ? (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {/* NO2 Validation */}
              <Card className="glass">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                    NO2 Model — Test Set Performance
                  </CardTitle>
                  <p className="text-xs text-secondary">t+1h horizon (best model)</p>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <p className="text-[10px] text-muted-foreground mb-1">R2</p>
                      <span className="text-2xl font-bold text-emerald-400">{vc.NO2.test.r2}</span>
                    </div>
                    <div>
                      <p className="text-[10px] text-muted-foreground mb-1">MAE (ppb)</p>
                      <span className="text-2xl font-bold text-emerald-400">{vc.NO2.test.mae}</span>
                    </div>
                    <div>
                      <p className="text-[10px] text-muted-foreground mb-1">RMSE (ppb)</p>
                      <span className="text-2xl font-bold text-emerald-400">{vc.NO2.test.rmse}</span>
                    </div>
                  </div>
                  <div className="mt-4 rounded-lg bg-muted/30 p-3">
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <CheckCircle2 className="h-3 w-3 text-emerald-400" />
                      Validated on {t.val_years} data, tested on {t.test_years}
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* O3 Validation */}
              <Card className="glass">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <CheckCircle2 className="h-4 w-4 text-sky-400" />
                    O3 Model — Test Set Performance
                  </CardTitle>
                  <p className="text-xs text-secondary">t+1h horizon (best model)</p>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <p className="text-[10px] text-muted-foreground mb-1">R2</p>
                      <span className="text-2xl font-bold text-sky-400">{vc.O3.test.r2}</span>
                    </div>
                    <div>
                      <p className="text-[10px] text-muted-foreground mb-1">MAE (ppb)</p>
                      <span className="text-2xl font-bold text-sky-400">{vc.O3.test.mae}</span>
                    </div>
                    <div>
                      <p className="text-[10px] text-muted-foreground mb-1">RMSE (ppb)</p>
                      <span className="text-2xl font-bold text-sky-400">{vc.O3.test.rmse}</span>
                    </div>
                  </div>
                  <div className="mt-4 rounded-lg bg-muted/30 p-3">
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <CheckCircle2 className="h-3 w-3 text-sky-400" />
                      Validated on {t.val_years} data, tested on {t.test_years}
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* HCHO Validation */}
              {vc.HCHO ? (
                <Card className="glass">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-base">
                      <CheckCircle2 className="h-4 w-4 text-violet-400" />
                      HCHO Model — Test Set Performance
                    </CardTitle>
                    <p className="text-xs text-secondary">t+1h horizon (µmol/m²)</p>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-3 gap-4">
                      <div>
                        <p className="text-[10px] text-muted-foreground mb-1">R2</p>
                        <span className="text-2xl font-bold text-violet-400">{vc.HCHO.test.r2}</span>
                      </div>
                      <div>
                        <p className="text-[10px] text-muted-foreground mb-1">MAE (µmol/m²)</p>
                        <span className="text-2xl font-bold text-violet-400">{vc.HCHO.test.mae}</span>
                      </div>
                      <div>
                        <p className="text-[10px] text-muted-foreground mb-1">RMSE (µmol/m²)</p>
                        <span className="text-2xl font-bold text-violet-400">{vc.HCHO.test.rmse}</span>
                      </div>
                    </div>
                    <div className="mt-4 rounded-lg bg-muted/30 p-3">
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <CheckCircle2 className="h-3 w-3 text-violet-400" />
                        Validated on {t.hcho_val_years ?? t.val_years} data, tested on {t.hcho_test_years ?? t.test_years}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ) : null}
            </div>
          ) : (
            <Card className="glass">
              <CardContent className="py-8 text-center text-sm text-muted-foreground">
                Validation data not available.
              </CardContent>
            </Card>
          )}

          {/* Accuracy of these predictions against the project database */}
          {dbVal && dbRows.length > 0 ? (
            <Card className="glass">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <Database className="h-4 w-4 text-secondary" />
                  Accuracy vs the project database
                </CardTitle>
                <p className="text-xs text-secondary">
                  {dbVal.database.extract} · {dbVal.database.rows.toLocaleString()} rows ·{" "}
                  {dbVal.database.min_date} → {dbVal.database.max_date} · joined with the model output on
                  identical site + timestamp ({dbVal.targets.NO2?.overlap.rows.toLocaleString() ?? 0} matches,
                  {dbVal.targets.NO2?.overlap.start ?? "?"} → {dbVal.targets.NO2?.overlap.end ?? "?"})
                </p>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-border">
                        <th className="px-2 py-1.5 text-left font-medium text-muted-foreground">Target</th>
                        <th className="px-2 py-1.5 text-left font-medium text-muted-foreground">Scored against</th>
                        <th className="px-2 py-1.5 text-right font-medium text-muted-foreground">R²</th>
                        <th className="px-2 py-1.5 text-right font-medium text-muted-foreground">MAE</th>
                        <th className="px-2 py-1.5 text-right font-medium text-muted-foreground">RMSE</th>
                        <th className="px-2 py-1.5 text-right font-medium text-muted-foreground">Bias</th>
                        <th className="px-2 py-1.5 text-right font-medium text-muted-foreground">n</th>
                      </tr>
                    </thead>
                    <tbody>
                      {dbRows.map((row) => (
                        <tr key={row.id} className="border-b border-border/50 hover:bg-muted/30">
                          <td className={`px-2 py-1.5 font-medium ${row.target === "NO2" ? "text-emerald-400" : row.target === "HCHO" ? "text-violet-400" : "text-sky-400"}`}>
                            {row.target}
                          </td>
                          <td className="px-2 py-1.5 text-muted-foreground">{row.basis}</td>
                          <td
                            className={`px-2 py-1.5 text-right font-mono ${
                              row.score.r2 == null ? "" : row.score.r2 < 0 ? "text-red-400" : "text-emerald-400"
                            }`}
                          >
                            {row.score.r2 == null ? "—" : row.score.r2.toFixed(3)}
                          </td>
                          <td className="px-2 py-1.5 text-right font-mono">{row.score.mae ?? "—"}</td>
                          <td className="px-2 py-1.5 text-right font-mono">{row.score.rmse ?? "—"}</td>
                          <td className="px-2 py-1.5 text-right font-mono">{row.score.bias ?? "—"}</td>
                          <td className="px-2 py-1.5 text-right font-mono text-muted-foreground">
                            {row.score.n.toLocaleString()}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="rounded-lg border border-border/60 bg-muted/30 p-3 text-xs leading-relaxed text-muted-foreground">
                  <p className="mb-2 text-foreground">
                    Row 1 is the number the cards above report. Row 2 scores the
                    <em> identical predictions</em> against the observations in the project database. A
                    negative R² means the forecast does not track that series at all.
                  </p>
                  <p>
                    The two sources disagree (correlation{" "}
                    <span className="font-mono text-foreground">
                      {dbVal.targets.NO2?.source_agreement.correlation ?? "—"}
                    </span>{" "}
                    for NO₂,{" "}
                    <span className="font-mono text-foreground">
                      {dbVal.targets.O3?.source_agreement.correlation ?? "—"}
                    </span>{" "}
                    for O₃ at the same site and hour) because the model was trained on the retired
                    <code className="px-1">site_*_train_data.csv</code> schema. Its{" "}
                    <code className="px-1">*_forecast</code> NWP inputs — about a quarter of the NO₂
                    model's gain — have no counterpart in{" "}
                    <code className="px-1">vaayu_final_v3.csv</code>, so the model cannot be re-run on the
                    database. Re-validating properly means joining those forecast columns back in for all
                    7 sites. HCHO (EXP_11) is the exception: it was trained directly on this database,
                    so its two rows should agree. Full write-up: <code className="px-1">models/ACCURACY.md</code>.
                  </p>
                </div>
              </CardContent>
            </Card>
          ) : null}
        </TabsContent>

        {/* ═══════════════ METHODOLOGY TAB ═══════════════ */}
        <TabsContent value="methodology" className="space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            {/* Feature Importance */}
            <Card className="glass">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <Layers className="h-4 w-4 text-secondary" />
                  Feature Importance — t+1h Models
                </CardTitle>
                <p className="text-xs text-secondary">
                  Top 10 features by XGBoost gain for NO2, O3 and HCHO at horizon 0
                </p>
              </CardHeader>
              <CardContent>
                {fi.NO2.length > 0 ? (
                  <div className="space-y-4">
                    <div>
                      <h4 className="mb-2 text-xs font-semibold text-emerald-400">NO2 Model</h4>
                      <div className="space-y-2">
                        {fi.NO2.map((f, i) => (
                          <div key={f.feature} className="flex items-center gap-3">
                            <span className="w-4 text-right text-[10px] text-muted-foreground">{i + 1}</span>
                            <span className="w-40 truncate text-xs font-medium">{f.feature}</span>
                            <div className="flex-1">
                              <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
                                <div
                                  className="h-full rounded-full bg-emerald-500 transition-all duration-700"
                                  style={{ width: `${f.importance}%` }}
                                />
                              </div>
                            </div>
                            <span className="w-12 text-right text-xs font-semibold text-emerald-400">{f.importance}%</span>
                          </div>
                        ))}
                      </div>
                    </div>
                    {fi.O3.length > 0 && (
                      <div>
                        <h4 className="mb-2 text-xs font-semibold text-sky-400">O3 Model</h4>
                        <div className="space-y-2">
                          {fi.O3.map((f, i) => (
                            <div key={f.feature} className="flex items-center gap-3">
                              <span className="w-4 text-right text-[10px] text-muted-foreground">{i + 1}</span>
                              <span className="w-40 truncate text-xs font-medium">{f.feature}</span>
                              <div className="flex-1">
                                <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
                                  <div
                                    className="h-full rounded-full bg-sky-500 transition-all duration-700"
                                    style={{ width: `${f.importance}%` }}
                                  />
                                </div>
                              </div>
                              <span className="w-12 text-right text-xs font-semibold text-sky-400">{f.importance}%</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    {(fi.HCHO?.length ?? 0) > 0 && (
                      <div>
                        <h4 className="mb-2 text-xs font-semibold text-violet-400">HCHO Model</h4>
                        <div className="space-y-2">
                          {fi.HCHO?.map((f, i) => (
                            <div key={f.feature} className="flex items-center gap-3">
                              <span className="w-4 text-right text-[10px] text-muted-foreground">{i + 1}</span>
                              <span className="w-40 truncate text-xs font-medium">{f.feature}</span>
                              <div className="flex-1">
                                <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
                                  <div
                                    className="h-full rounded-full bg-violet-500 transition-all duration-700"
                                    style={{ width: `${f.importance}%` }}
                                  />
                                </div>
                              </div>
                              <span className="w-12 text-right text-xs font-semibold text-violet-400">{f.importance}%</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <p className="text-xs text-muted-foreground">Feature importance data not available.</p>
                )}
              </CardContent>
            </Card>

            {/* Hyperparameters + Training Config */}
            <div className="space-y-6">
              <Card className="glass">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Zap className="h-4 w-4 text-amber-400" />
                    Hyperparameters
                  </CardTitle>
                  <p className="text-xs text-secondary">XGBoost EXP_09v2 configuration</p>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {Object.entries(hp).map(([key, val]) => (
                      <div key={key} className="flex items-center justify-between gap-2">
                        <span className="text-xs text-muted-foreground">{key.replace(/_/g, " ")}</span>
                        <Badge variant="secondary" className="shrink-0 text-[10px]">{val}</Badge>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              <Card className="glass">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Database className="h-4 w-4 text-purple-400" />
                    Training Configuration
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-muted-foreground">Training Period</span>
                      <Badge variant="secondary" className="text-[10px]">{t.train_years}</Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-muted-foreground">Validation Period</span>
                      <Badge variant="secondary" className="text-[10px]">{t.val_years}</Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-muted-foreground">Test Period</span>
                      <Badge variant="secondary" className="text-[10px]">{t.test_years}</Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-muted-foreground">Monitoring Sites</span>
                      <Badge variant="secondary" className="text-[10px]">{t.sites} stations</Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-muted-foreground">Forecast Horizons</span>
                      <Badge variant="secondary" className="text-[10px]">{t.horizons} steps</Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-muted-foreground">Feature Count</span>
                      <Badge variant="secondary" className="text-[10px]">{t.features} features</Badge>
                    </div>
                    {t.features_hcho && (
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-muted-foreground">Feature Count (HCHO)</span>
                        <Badge variant="secondary" className="text-[10px]">{t.features_hcho} features</Badge>
                      </div>
                    )}
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-muted-foreground">Target Variables</span>
                      <Badge variant="secondary" className="text-[10px]">NO2, O3{m.t1h.HCHO ? ", HCHO" : ""}</Badge>
                    </div>
                    {m.t1h.NO2.best_iter && (
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-muted-foreground">Best Iteration (NO2)</span>
                        <Badge variant="secondary" className="text-[10px]">{m.t1h.NO2.best_iter}</Badge>
                      </div>
                    )}
                    {m.t1h.O3.best_iter && (
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-muted-foreground">Best Iteration (O3)</span>
                        <Badge variant="secondary" className="text-[10px]">{m.t1h.O3.best_iter}</Badge>
                      </div>
                    )}
                    {m.t1h.HCHO?.best_iter && (
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-muted-foreground">Best Iteration (HCHO)</span>
                        <Badge variant="secondary" className="text-[10px]">{m.t1h.HCHO.best_iter}</Badge>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </section>
  );
}

const TARGET_COLORS: Record<string, string> = {
  NO2: "text-emerald-400",
  O3: "text-sky-400",
  HCHO: "text-violet-400",
};

function ModelCard({ block }: { block: ModelBlock }) {
  const targets = (["NO2", "O3", "HCHO"] as const).filter(
    (t) => block.per_target[t] || block.horizon_metrics[t]
  );

  return (
    <Card className={`glass relative overflow-hidden ${block.active ? "ring-1 ring-emerald-500/50" : ""}`}>
      {block.active && (
        <div className="absolute right-3 top-3">
          <Badge className="bg-emerald-500/15 text-emerald-400">
            <Award className="mr-1 h-3 w-3" /> ACTIVE
          </Badge>
        </div>
      )}
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-base">
          <Cpu className="h-4 w-4 text-secondary" />
          {block.name}
        </CardTitle>
        <p className="text-xs text-secondary">
          {block.available
            ? `Rolling ${7}-day RMSE: ${fmt(block.rolling_rmse, 2)}`
            : block.reason}
        </p>
      </CardHeader>
      {block.available && (
        <CardContent className="space-y-4">
          {/* Per-target headline metrics (t+1h pooled across horizons) */}
          <div className="grid grid-cols-3 gap-2">
            {targets.map((t) => {
              const all = block.per_target[t]?.all;
              return (
                <div key={t} className="rounded-lg bg-muted/30 p-2">
                  <p className={`text-[10px] font-medium ${TARGET_COLORS[t]}`}>{t}</p>
                  <p className="text-[10px] text-muted-foreground">
                    R² {fmt(all?.r2)} · RMSE {fmt(all?.rmse, 2)} · MAE {fmt(all?.mae, 2)}
                  </p>
                  <p className="text-[10px] text-muted-foreground">
                    Rolling RMSE {fmt(block.rolling_by_target[t], 2)}
                  </p>
                </div>
              );
            })}
          </div>

          {/* Strongest horizons */}
          <div className="space-y-1">
            {targets.map((t) => {
              const hs = block.strongest_horizons[t];
              if (!hs?.length) return null;
              return (
                <p key={t} className="text-[10px] text-muted-foreground">
                  <span className={`font-medium ${TARGET_COLORS[t]}`}>{t}</span> strongest at{" "}
                  {hs.map((h) => `t+${h + 1}h`).join(", ")}
                </p>
              );
            })}
          </div>

          {/* Per-horizon table */}
          {targets.map((t) => {
            const rows = block.horizon_metrics[t];
            if (!rows?.length) return null;
            const shown = HORIZON_ROWS.map((i) => rows[i]).filter(Boolean);
            return (
              <div key={t}>
                <p className={`mb-1 text-[10px] font-semibold ${TARGET_COLORS[t]}`}>{t} per-horizon</p>
                <div className="overflow-x-auto">
                  <table className="w-full text-[10px]">
                    <thead>
                      <tr className="border-b border-border">
                        <th className="px-1 py-1 text-left font-medium text-muted-foreground">Horizon</th>
                        <th className="px-1 py-1 text-right font-medium text-muted-foreground">R²</th>
                        <th className="px-1 py-1 text-right font-medium text-muted-foreground">RMSE</th>
                        <th className="px-1 py-1 text-right font-medium text-muted-foreground">MAE</th>
                      </tr>
                    </thead>
                    <tbody>
                      {shown.map((r) => (
                        <tr key={r.horizon} className="border-b border-border/50">
                          <td className="px-1 py-1">t+{r.horizon + 1}h</td>
                          <td className="px-1 py-1 text-right font-mono">{fmt(r.r2)}</td>
                          <td className="px-1 py-1 text-right font-mono">{fmt(r.rmse, 2)}</td>
                          <td className="px-1 py-1 text-right font-mono">{fmt(r.mae, 2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            );
          })}
        </CardContent>
      )}
    </Card>
  );
}
