"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, CheckCircle2 } from "lucide-react";

import { AiInsightsPanel } from "@/components/dashboard/ai-insights-panel";
import { DesktopSidebar, type DashboardView } from "@/components/dashboard/app-sidebar";
import { CustomersView } from "@/components/dashboard/customers-view";
import { DashboardToolbar } from "@/components/dashboard/dashboard-toolbar";
import { InvoiceDetailPanel } from "@/components/dashboard/invoice-detail-panel";
import { PaidInvoicesTable } from "@/components/dashboard/paid-invoices-table";
import { PrioritizedTable } from "@/components/dashboard/prioritized-table";
import { SummaryCards } from "@/components/dashboard/summary-cards";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";
import { DEFAULT_CUTOFF } from "@/lib/formatters";
import type {
  Customer,
  DashboardSummary,
  Interaction,
  InteractionCreateInput,
  Invoice,
  InvoiceCreateInput,
  InvoiceUpdateInput,
  PaymentCreateInput,
  PaymentPromise,
  Prediction,
  PredictionDailyItem,
  PredictionHistoryItem,
  PromiseCreateInput,
  PromiseUpdateInput,
  PrioritizedInvoice,
  RecalculateResult,
  SimulationChange,
  Segment
} from "@/lib/types";

const DEFAULT_PAGE_SIZE = 15;
const PRIORITIZED_LIMIT = 200;
type ScoreFilter = "todos" | "critico" | "alto" | "medio" | "bajo";
type QueueLoadResult = {
  summary: DashboardSummary;
  preventive: PrioritizedInvoice[];
  overdue: PrioritizedInvoice[];
  paid: PrioritizedInvoice[];
};

type RecalculateOptions = {
  keepSelected?: boolean;
  preferredFacturaId?: string;
  queueView?: DashboardView;
};

export function DashboardShell() {
  const [fechaCorte, setFechaCorte] = useState(DEFAULT_CUTOFF);
  const [activeView, setActiveView] = useState<DashboardView>("preventive");
  const [dashboard, setDashboard] = useState<DashboardSummary | null>(null);
  const [preventiveRows, setPreventiveRows] = useState<PrioritizedInvoice[]>([]);
  const [overdueRows, setOverdueRows] = useState<PrioritizedInvoice[]>([]);
  const [paidRows, setPaidRows] = useState<PrioritizedInvoice[]>([]);
  const [selected, setSelected] = useState<PrioritizedInvoice | null>(null);
  const [invoice, setInvoice] = useState<Invoice | null>(null);
  const [customer, setCustomer] = useState<Customer | null>(null);
  const [segment, setSegment] = useState<Segment | null>(null);
  const [customers, setCustomers] = useState<Customer[]>([]);
  const [selectedCustomer, setSelectedCustomer] = useState<Customer | null>(null);
  const [selectedCustomerSegment, setSelectedCustomerSegment] = useState<Segment | null>(null);
  const [interactions, setInteractions] = useState<Interaction[]>([]);
  const [promises, setPromises] = useState<PaymentPromise[]>([]);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [predictionHistory, setPredictionHistory] = useState<PredictionHistoryItem[]>([]);
  const [predictionDaily, setPredictionDaily] = useState<PredictionDailyItem[]>([]);
  const [batchResult, setBatchResult] = useState<RecalculateResult | null>(null);
  const [simulationChanges, setSimulationChanges] = useState<SimulationChange[]>([]);
  const [query, setQuery] = useState("");
  const [customerQuery, setCustomerQuery] = useState("");
  const [scoreFilter, setScoreFilter] = useState<ScoreFilter>("todos");
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE);
  const [loading, setLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [createdInvoiceNotice, setCreatedInvoiceNotice] = useState<string | null>(null);
  const [detailVersion, setDetailVersion] = useState(0);
  const refreshSequence = useRef(0);

  const activeRows = useMemo(() => {
    if (activeView === "overdue") return overdueRows;
    if (activeView === "paid") return paidRows;
    return preventiveRows;
  }, [activeView, overdueRows, paidRows, preventiveRows]);

  const filteredRows = useMemo(() => {
    const normalized = query.trim().toLowerCase();

    return activeRows.filter((row) => {
      const matchesQuery =
        !normalized ||
        [
          row.factura_id,
          row.cliente_id,
          row.cliente_nombre,
          row.sector,
          row.predicted_label_usuario ?? "",
          row.accion_sugerida ?? ""
        ]
          .join(" ")
          .toLowerCase()
          .includes(normalized);

      const score = row.priority_score_0_100;
      const matchesScore =
        scoreFilter === "todos" ||
        (score !== null &&
          score !== undefined &&
          ((scoreFilter === "critico" && score >= 80) ||
            (scoreFilter === "alto" && score >= 60 && score < 80) ||
            (scoreFilter === "medio" && score >= 40 && score < 60) ||
            (scoreFilter === "bajo" && score < 40)));

      return matchesQuery && matchesScore;
    });
  }, [activeRows, query, scoreFilter]);

  const filteredCustomers = useMemo(() => {
    const normalized = customerQuery.trim().toLowerCase();
    if (!normalized) return customers;
    return customers.filter((item) =>
      [item.cliente_id, item.nombre, item.sector].join(" ").toLowerCase().includes(normalized)
    );
  }, [customerQuery, customers]);

  const totalPages = Math.max(1, Math.ceil(filteredRows.length / pageSize));
  const paginatedRows = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return filteredRows.slice(start, start + pageSize);
  }, [currentPage, filteredRows, pageSize]);
  const selectedFacturaId = selected?.factura_id;
  const selectedClienteId = selected?.cliente_id;

  function rowsForView(
    view: DashboardView,
    preventive: PrioritizedInvoice[],
    overdue: PrioritizedInvoice[],
    paid: PrioritizedInvoice[]
  ) {
    if (view === "overdue") return overdue;
    if (view === "paid") return paid;
    return preventive;
  }

  function viewForInvoiceAtCutoff(item: Invoice): DashboardView {
    if (item.fecha_pago_real && item.fecha_pago_real <= fechaCorte) return "paid";
    if (item.fecha_vencimiento < fechaCorte) return "overdue";
    return "preventive";
  }

  async function loadSummaryAndQueue(
    keepSelected = true,
    preferredFacturaId?: string,
    queueView: DashboardView = activeView,
    cutoffDate = fechaCorte,
    refreshId?: number
  ): Promise<QueueLoadResult | null> {
    setError(null);
    const [summary, preventive, overdue, paid] = await Promise.all([
      api.dashboard(cutoffDate),
      api.prioritized(PRIORITIZED_LIMIT, cutoffDate, "preventive"),
      api.prioritized(PRIORITIZED_LIMIT, cutoffDate, "overdue"),
      api.prioritized(PRIORITIZED_LIMIT, cutoffDate, "paid")
    ]);
    if (refreshId !== undefined && refreshId !== refreshSequence.current) return null;

    setDashboard(summary);
    setPreventiveRows(preventive);
    setOverdueRows(overdue);
    setPaidRows(paid);
    const queue = rowsForView(queueView, preventive, overdue, paid);
    const allRows = [...preventive, ...overdue, ...paid];
    setSelected((current) => {
      if (preferredFacturaId) {
        return allRows.find((row) => row.factura_id === preferredFacturaId) ?? queue[0] ?? null;
      }
      if (!keepSelected) return queue[0] ?? null;
      if (!current) return queue[0] ?? null;
      return queue.find((row) => row.factura_id === current.factura_id) ?? queue[0] ?? null;
    });
    return { summary, preventive, overdue, paid };
  }

  async function loadCustomers(keepSelected = true) {
    const data = await api.customers(200, 0);
    setCustomers(data);
    setSelectedCustomer((current) => {
      if (!keepSelected) return data[0] ?? null;
      if (!current) return data[0] ?? null;
      return data.find((item) => item.cliente_id === current.cliente_id) ?? data[0] ?? null;
    });
  }

  async function initializeData() {
    refreshSequence.current += 1;
    setLoading(true);
    setError(null);
    setCreatedInvoiceNotice(null);
    try {
      await api.initDb();
      setBatchResult(null);
      setPrediction(null);
      await recalculate({ keepSelected: false });
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo inicializar la base.");
    } finally {
      setLoading(false);
    }
  }

  async function recalculate(options: RecalculateOptions = {}) {
    const { keepSelected = true, preferredFacturaId, queueView = activeView } = options;
    const refreshId = ++refreshSequence.current;
    const cutoffDate = fechaCorte;
    setLoading(true);
    setError(null);
    if (!preferredFacturaId) setCreatedInvoiceNotice(null);
    try {
      const before = [...preventiveRows, ...overdueRows].map((row, index) => ({
        factura_id: row.factura_id,
        priority_score_0_100: row.priority_score_0_100,
        accion_sugerida: row.accion_sugerida,
        position: index + 1
      }));
      const result = await api.recalculate(cutoffDate, PRIORITIZED_LIMIT);
      if (refreshId !== refreshSequence.current) return;
      setBatchResult(result);
      setPrediction(null);
      const loaded = await loadSummaryAndQueue(keepSelected, preferredFacturaId, queueView, cutoffDate, refreshId);
      if (!loaded) return;
      const beforeById = new Map(before.map((item) => [item.factura_id, item]));
      const changes: SimulationChange[] = [...loaded.preventive, ...loaded.overdue]
        .slice(0, 25)
        .map<SimulationChange | null>((row, index) => {
          const previous = beforeById.get(row.factura_id);
          if (!previous) return null;
          const previousScore = previous.priority_score_0_100;
          const nextScore = row.priority_score_0_100;
          return {
            factura_id: row.factura_id,
            score_delta:
              previousScore === null || previousScore === undefined || nextScore === null || nextScore === undefined
                ? null
                : nextScore - previousScore,
            accion_anterior: previous.accion_sugerida,
            accion_nueva: row.accion_sugerida,
            posicion_delta: previous.position - (index + 1)
          };
        })
        .filter((item): item is SimulationChange => item !== null)
        .filter((item) => item.score_delta !== 0 || item.accion_anterior !== item.accion_nueva || item.posicion_delta !== 0)
        .slice(0, 5);
      setSimulationChanges(changes);
      setDetailVersion((value) => value + 1);
    } catch (err) {
      if (refreshId === refreshSequence.current) {
        setError(err instanceof Error ? err.message : "No se pudo recalcular la cartera.");
      }
    } finally {
      if (refreshId === refreshSequence.current) setLoading(false);
    }
  }

  async function scoreSelected() {
    if (!selected) return;
    setDetailLoading(true);
    setError(null);
    try {
      const result = await api.score(selected.factura_id, fechaCorte, true);
      await loadSummaryAndQueue(true);
      setPrediction(result);
      const [history, daily] = await Promise.all([
        api.predictionHistory(selected.factura_id, fechaCorte),
        api.predictionDaily(selected.factura_id, fechaCorte)
      ]);
      setPredictionHistory(history);
      setPredictionDaily(daily);
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo actualizar la prediccion.");
    } finally {
      setDetailLoading(false);
    }
  }

  async function createInvoice(payload: InvoiceCreateInput) {
    setLoading(true);
    setError(null);
    setCreatedInvoiceNotice(null);
    setBatchResult(null);
    setSimulationChanges([]);
    try {
      const created = await api.createInvoice(payload);
      const targetView = viewForInvoiceAtCutoff(created);
      let scoreWarning: string | null = null;
      let scoredPrediction: Prediction | null = null;
      if (!["pagada", "anulada", "castigada"].includes(created.estado_factura) && created.fecha_emision <= fechaCorte) {
        try {
          scoredPrediction = await api.score(created.factura_id, fechaCorte, true);
        } catch (err) {
          scoreWarning = err instanceof Error ? err.message : "Factura creada sin prediccion calculada.";
        }
      }
      setQuery(created.factura_id);
      setScoreFilter("todos");
      setCurrentPage(1);
      setActiveView(targetView);
      setPrediction(scoredPrediction);
      await loadSummaryAndQueue(true, created.factura_id, targetView);
      setDetailVersion((value) => value + 1);
      setCreatedInvoiceNotice(`Factura ${created.factura_id} creada y seleccionada.`);
      if (scoreWarning) {
        setError(`Factura creada, pero no se pudo calcular la prediccion: ${scoreWarning}`);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "No se pudo crear la factura.";
      setError(message);
      throw new Error(message);
    } finally {
      setLoading(false);
    }
  }

  async function createInteraction(payload: InteractionCreateInput) {
    setDetailLoading(true);
    setError(null);
    setBatchResult(null);
    setSimulationChanges([]);
    try {
      await api.createInteraction(payload);
      let scoreWarning: string | null = null;
      let scoredPrediction: Prediction | null = null;
      try {
        scoredPrediction = await api.score(payload.factura_id, fechaCorte, true);
      } catch (err) {
        scoreWarning = err instanceof Error ? err.message : "Gestion registrada sin prediccion recalculada.";
      }
      setPrediction(scoredPrediction);
      await loadSummaryAndQueue(true, payload.factura_id, activeView);
      setDetailVersion((value) => value + 1);
      if (scoreWarning) {
        setError(`Gestion registrada, pero no se pudo recalcular la prediccion: ${scoreWarning}`);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "No se pudo registrar la gestion.";
      setError(message);
      throw new Error(message);
    } finally {
      setDetailLoading(false);
    }
  }

  async function createPromise(payload: PromiseCreateInput) {
    if (!selected) return;
    setDetailLoading(true);
    setError(null);
    setBatchResult(null);
    setSimulationChanges([]);
    try {
      await api.createPromise(payload);
      let scoreWarning: string | null = null;
      let scoredPrediction: Prediction | null = null;
      try {
        scoredPrediction = await api.score(selected.factura_id, fechaCorte, true);
      } catch (err) {
        scoreWarning = err instanceof Error ? err.message : "Promesa registrada sin prediccion recalculada.";
      }
      setPrediction(scoredPrediction);
      await loadSummaryAndQueue(true, selected.factura_id, activeView);
      setDetailVersion((value) => value + 1);
      if (scoreWarning) {
        setError(`Promesa registrada, pero no se pudo recalcular la prediccion: ${scoreWarning}`);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "No se pudo registrar la promesa.";
      setError(message);
      throw new Error(message);
    } finally {
      setDetailLoading(false);
    }
  }

  async function updateInvoice(facturaId: string, payload: InvoiceUpdateInput) {
    setDetailLoading(true);
    setError(null);
    setBatchResult(null);
    setSimulationChanges([]);
    try {
      const updated = await api.updateInvoice(facturaId, payload);
      let scoreWarning: string | null = null;
      let scoredPrediction: Prediction | null = null;
      const shouldScore =
        updated.fecha_emision <= fechaCorte && !["pagada", "anulada", "castigada"].includes(updated.estado_factura);
      if (shouldScore) {
        try {
          scoredPrediction = await api.score(updated.factura_id, fechaCorte, true);
        } catch (err) {
          scoreWarning = err instanceof Error ? err.message : "Factura actualizada sin prediccion recalculada.";
        }
      }
      const targetView = viewForInvoiceAtCutoff(updated);
      setActiveView(targetView);
      setPrediction(scoredPrediction);
      await loadSummaryAndQueue(true, updated.factura_id, targetView);
      setDetailVersion((value) => value + 1);
      if (scoreWarning) setError(`Factura actualizada, pero no se pudo recalcular la prediccion: ${scoreWarning}`);
    } catch (err) {
      const message = err instanceof Error ? err.message : "No se pudo actualizar la factura.";
      setError(message);
      throw new Error(message);
    } finally {
      setDetailLoading(false);
    }
  }

  async function updatePromise(promesaId: string, payload: PromiseUpdateInput) {
    if (!selected) return;
    setDetailLoading(true);
    setError(null);
    setBatchResult(null);
    setSimulationChanges([]);
    try {
      await api.updatePromise(promesaId, payload);
      let scoreWarning: string | null = null;
      let scoredPrediction: Prediction | null = null;
      try {
        scoredPrediction = await api.score(selected.factura_id, fechaCorte, true);
      } catch (err) {
        scoreWarning = err instanceof Error ? err.message : "Promesa actualizada sin prediccion recalculada.";
      }
      setPrediction(scoredPrediction);
      await loadSummaryAndQueue(true, selected.factura_id, activeView);
      setDetailVersion((value) => value + 1);
      if (scoreWarning) setError(`Promesa actualizada, pero no se pudo recalcular la prediccion: ${scoreWarning}`);
    } catch (err) {
      const message = err instanceof Error ? err.message : "No se pudo actualizar la promesa.";
      setError(message);
      throw new Error(message);
    } finally {
      setDetailLoading(false);
    }
  }

  async function registerPayment(payload: PaymentCreateInput) {
    setDetailLoading(true);
    setError(null);
    setBatchResult(null);
    setSimulationChanges([]);
    try {
      const paidInvoice = await api.registerPayment(payload);
      setActiveView("paid");
      setPrediction(null);
      await loadSummaryAndQueue(true, paidInvoice.factura_id, "paid");
      setDetailVersion((value) => value + 1);
    } catch (err) {
      const message = err instanceof Error ? err.message : "No se pudo registrar el pago.";
      setError(message);
      throw new Error(message);
    } finally {
      setDetailLoading(false);
    }
  }

  useEffect(() => {
    recalculate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fechaCorte]);

  useEffect(() => {
    setCurrentPage(1);
  }, [pageSize, query, scoreFilter, activeView]);

  useEffect(() => {
    if (currentPage > totalPages) setCurrentPage(totalPages);
  }, [currentPage, totalPages]);

  useEffect(() => {
    if (activeView === "customers" && customers.length === 0) {
      loadCustomers().catch((err) => {
        setError(err instanceof Error ? err.message : "No se pudo cargar clientes.");
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeView]);

  useEffect(() => {
    if (!selectedCustomer) {
      setSelectedCustomerSegment(null);
      return;
    }
    api.segment(selectedCustomer.cliente_id)
      .then(setSelectedCustomerSegment)
      .catch((err) => {
        setSelectedCustomerSegment(null);
        setError(err instanceof Error ? err.message : "No se pudo cargar segmento del cliente.");
      });
  }, [selectedCustomer]);

  useEffect(() => {
    if (["preventive", "overdue", "paid"].includes(activeView)) {
      setSelected((current) => {
        if (current && activeRows.some((row) => row.factura_id === current.factura_id)) return current;
        return activeRows[0] ?? null;
      });
    }
  }, [activeRows, activeView]);

  useEffect(() => {
    if (!selectedFacturaId || !selectedClienteId) {
      setInvoice(null);
      setCustomer(null);
      setSegment(null);
      setInteractions([]);
      setPromises([]);
      setPredictionHistory([]);
      setPredictionDaily([]);
      return;
    }

    let cancelled = false;
    setDetailLoading(true);
    setPrediction(null);
    setInvoice(null);
    setCustomer(null);
    setSegment(null);
    setInteractions([]);
    setPromises([]);
    setPredictionHistory([]);
    setPredictionDaily([]);

    Promise.all([
      api.invoice(selectedFacturaId, fechaCorte),
      api.customer(selectedClienteId),
      api.segment(selectedClienteId),
      api.interactions(selectedFacturaId, fechaCorte),
      api.invoicePromises(selectedFacturaId, fechaCorte),
      api.predictionHistory(selectedFacturaId, fechaCorte),
      api.predictionDaily(selectedFacturaId, fechaCorte)
    ])
      .then(([invoiceData, customerData, segmentData, interactionsData, promisesData, predictionHistoryData, predictionDailyData]) => {
        if (cancelled) return;
        setInvoice(invoiceData);
        setCustomer(customerData);
        setSegment(segmentData);
        setInteractions(interactionsData);
        setPromises(promisesData);
        setPredictionHistory(predictionHistoryData);
        setPredictionDaily(predictionDailyData);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "No se pudo cargar el detalle.");
      })
      .finally(() => {
        if (!cancelled) setDetailLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [detailVersion, fechaCorte, selectedClienteId, selectedFacturaId]);

  return (
    <div className="min-h-screen bg-muted/35">
      <DesktopSidebar activeView={activeView} onViewChange={setActiveView} />
      <main className="px-4 py-5 md:px-6 lg:ml-64 lg:px-8">
        <div className="mx-auto max-w-[1520px]">
          <DashboardToolbar
            fechaCorte={fechaCorte}
            loading={loading}
            onFechaCorteChange={setFechaCorte}
            onInitialize={initializeData}
            onRecalculate={() => recalculate()}
            onCreateInvoice={createInvoice}
            activeView={activeView}
            onViewChange={setActiveView}
          />

          {error && (
            <Alert variant="destructive" className="mb-4">
              <AlertTriangle />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {createdInvoiceNotice && (
            <Alert className="mb-4">
              <CheckCircle2 />
              <AlertDescription>{createdInvoiceNotice} El buscador quedo filtrado con ese codigo.</AlertDescription>
            </Alert>
          )}

          <SummaryCards dashboard={dashboard} />

          {activeView === "metrics" && <AiInsightsPanel rows={[...preventiveRows, ...overdueRows]} />}

          {batchResult && (
            <section className="mt-4 flex flex-wrap items-center gap-3 rounded-md border bg-background px-4 py-3 text-sm shadow-sm">
              <CheckCircle2 className="size-5 text-muted-foreground" />
              <span className="font-medium">{batchResult.total_evaluadas} facturas evaluadas</span>
              <span className="text-muted-foreground">Fecha de corte {batchResult.fecha_corte}</span>
              {batchResult.total_con_error > 0 && (
                <Badge variant="destructive">{batchResult.total_con_error} con error</Badge>
              )}
              {simulationChanges.map((item) => (
                <Badge key={item.factura_id} variant="outline">
                  {item.factura_id}: score {item.score_delta === null ? "s/d" : item.score_delta.toFixed(1)}, pos{" "}
                  {item.posicion_delta === null ? "s/d" : item.posicion_delta > 0 ? `+${item.posicion_delta}` : item.posicion_delta}
                </Badge>
              ))}
            </section>
          )}

          {activeView === "customers" ? (
            <div className="mt-5">
              <CustomersView
                customers={filteredCustomers}
                selectedCustomer={selectedCustomer}
                selectedSegment={selectedCustomerSegment}
                query={customerQuery}
                loading={loading}
                onQueryChange={setCustomerQuery}
                onSelect={setSelectedCustomer}
              />
            </div>
          ) : ["preventive", "overdue", "paid"].includes(activeView) ? (
            <section className="mt-5 grid gap-5 xl:grid-cols-[minmax(0,1fr)_440px]">
              {activeView === "paid" ? (
                <PaidInvoicesTable
                  rows={filteredRows}
                  query={query}
                  onQueryChange={setQuery}
                  selected={selected}
                  onSelect={setSelected}
                />
              ) : (
                <PrioritizedTable
                  rows={paginatedRows}
                  totalRows={filteredRows.length}
                  title={activeView === "overdue" ? "Vencidas al corte" : "Cola preventiva"}
                  emptyTitle={activeView === "overdue" ? "Sin facturas vencidas" : "Sin facturas preventivas"}
                  emptyDescription={
                    activeView === "overdue"
                      ? "No hay facturas abiertas y vencidas para la fecha de corte seleccionada."
                      : "No hay facturas abiertas no vencidas para la fecha de corte seleccionada."
                  }
                  mode={activeView === "overdue" ? "overdue" : "preventive"}
                  query={query}
                  scoreFilter={scoreFilter}
                  currentPage={currentPage}
                  totalPages={totalPages}
                  pageSize={pageSize}
                  selected={selected}
                  onQueryChange={setQuery}
                  onScoreFilterChange={setScoreFilter}
                  onPageChange={setCurrentPage}
                  onPageSizeChange={setPageSize}
                  onSelect={setSelected}
                />
              )}

              <InvoiceDetailPanel
                selected={selected}
                invoice={invoice}
                customer={customer}
                segment={segment}
                interactions={interactions}
                prediction={prediction}
                predictionHistory={predictionHistory}
                predictionDaily={predictionDaily}
                promises={promises}
                fechaCorte={fechaCorte}
                detailLoading={detailLoading}
                canScore={activeView !== "paid"}
                onScore={scoreSelected}
                onUpdateInvoice={updateInvoice}
                onCreateInteraction={createInteraction}
                onCreatePromise={createPromise}
                onUpdatePromise={updatePromise}
                onRegisterPayment={registerPayment}
              />
            </section>
          ) : null}
        </div>
      </main>
    </div>
  );
}
