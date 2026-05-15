"use client";

import { useEffect, useMemo, useState } from "react";
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
  Invoice,
  Prediction,
  PredictionDailyItem,
  PredictionHistoryItem,
  PrioritizedInvoice,
  RecalculateResult,
  Segment
} from "@/lib/types";

const DEFAULT_PAGE_SIZE = 15;
const PRIORITIZED_LIMIT = 200;
type ScoreFilter = "todos" | "critico" | "alto" | "medio" | "bajo";

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
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [predictionHistory, setPredictionHistory] = useState<PredictionHistoryItem[]>([]);
  const [predictionDaily, setPredictionDaily] = useState<PredictionDailyItem[]>([]);
  const [batchResult, setBatchResult] = useState<RecalculateResult | null>(null);
  const [query, setQuery] = useState("");
  const [customerQuery, setCustomerQuery] = useState("");
  const [scoreFilter, setScoreFilter] = useState<ScoreFilter>("todos");
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE);
  const [loading, setLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

  async function loadSummaryAndQueue(keepSelected = true) {
    setError(null);
    const [summary, preventive, overdue, paid] = await Promise.all([
      api.dashboard(fechaCorte),
      api.prioritized(PRIORITIZED_LIMIT, fechaCorte, "preventive"),
      api.prioritized(PRIORITIZED_LIMIT, fechaCorte, "overdue"),
      api.prioritized(PRIORITIZED_LIMIT, fechaCorte, "paid")
    ]);

    setDashboard(summary);
    setPreventiveRows(preventive);
    setOverdueRows(overdue);
    setPaidRows(paid);
    const queue = activeView === "overdue" ? overdue : activeView === "paid" ? paid : preventive;
    setSelected((current) => {
      if (!keepSelected) return queue[0] ?? null;
      if (!current) return queue[0] ?? null;
      return queue.find((row) => row.factura_id === current.factura_id) ?? queue[0] ?? null;
    });
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
    setLoading(true);
    setError(null);
    try {
      await api.initDb();
      setBatchResult(null);
      setPrediction(null);
      await loadSummaryAndQueue(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo inicializar la base.");
    } finally {
      setLoading(false);
    }
  }

  async function recalculate() {
    setLoading(true);
    setError(null);
    try {
      const result = await api.recalculate(fechaCorte, PRIORITIZED_LIMIT);
      setBatchResult(result);
      setPrediction(null);
      await loadSummaryAndQueue(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo recalcular la cartera.");
    } finally {
      setLoading(false);
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

  useEffect(() => {
    loadSummaryAndQueue().catch((err) => {
      setError(err instanceof Error ? err.message : "No se pudo conectar con el backend.");
    });
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
    setPredictionHistory([]);
    setPredictionDaily([]);

    Promise.all([
      api.invoice(selectedFacturaId, fechaCorte),
      api.customer(selectedClienteId),
      api.segment(selectedClienteId),
      api.interactions(selectedFacturaId, fechaCorte),
      api.predictionHistory(selectedFacturaId, fechaCorte),
      api.predictionDaily(selectedFacturaId, fechaCorte)
    ])
      .then(([invoiceData, customerData, segmentData, interactionsData, predictionHistoryData, predictionDailyData]) => {
        if (cancelled) return;
        setInvoice(invoiceData);
        setCustomer(customerData);
        setSegment(segmentData);
        setInteractions(interactionsData);
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
  }, [fechaCorte, selectedClienteId, selectedFacturaId]);

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
            onRecalculate={recalculate}
            activeView={activeView}
            onViewChange={setActiveView}
          />

          {error && (
            <Alert variant="destructive" className="mb-4">
              <AlertTriangle />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <SummaryCards dashboard={dashboard} />

          {(activeView === "summary" || activeView === "metrics") && (
            <AiInsightsPanel rows={activeView === "metrics" ? [...preventiveRows, ...overdueRows] : preventiveRows} />
          )}

          {batchResult && (
            <section className="mt-4 flex flex-wrap items-center gap-3 rounded-md border bg-background px-4 py-3 text-sm shadow-sm">
              <CheckCircle2 className="size-5 text-muted-foreground" />
              <span className="font-medium">{batchResult.total_evaluadas} facturas evaluadas</span>
              <span className="text-muted-foreground">Fecha de corte {batchResult.fecha_corte}</span>
              {batchResult.total_con_error > 0 && (
                <Badge variant="destructive">{batchResult.total_con_error} con error</Badge>
              )}
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
                fechaCorte={fechaCorte}
                detailLoading={detailLoading}
                canScore={activeView !== "paid"}
                onScore={scoreSelected}
              />
            </section>
          ) : null}
        </div>
      </main>
    </div>
  );
}
