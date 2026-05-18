import type {
  Customer,
  DashboardSummary,
  Interaction,
  InteractionCreated,
  InteractionCreateInput,
  Invoice,
  InvoiceCreateInput,
  InvoiceUpdateInput,
  PaymentCreateInput,
  Prediction,
  PredictionDailyItem,
  PredictionHistoryItem,
  PromiseCreated,
  PromiseCreateInput,
  PaymentPromise,
  PromiseUpdateInput,
  PrioritizedInvoice,
  RecalculateResult,
  Segment
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000/api/v1";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (init?.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers,
    cache: "no-store"
  });

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = typeof body.detail === "string" ? body.detail : response.statusText;
    throw new Error(detail);
  }

  return response.json() as Promise<T>;
}

export const api = {
  initDb: () => request<Record<string, number>>("/admin/init-db", { method: "POST" }),
  dashboard: (fechaCorte: string) =>
    request<DashboardSummary>(`/dashboard/summary?fecha_corte=${fechaCorte}`),
  prioritized: (limit = 50, fechaCorte?: string, estadoCorte?: "preventive" | "overdue" | "paid") =>
    request<PrioritizedInvoice[]>(
      `/invoices/prioritized?limit=${limit}${fechaCorte ? `&fecha_corte=${fechaCorte}` : ""}${
        estadoCorte ? `&estado_corte=${estadoCorte}` : ""
      }`
    ),
  recalculate: (fechaCorte: string, limit = 100) =>
    request<RecalculateResult>("/scoring/recalculate", {
      method: "POST",
      body: JSON.stringify({ fecha_corte: fechaCorte, limit, persist: true })
    }),
  invoice: (facturaId: string, fechaCorte?: string) =>
    request<Invoice>(`/invoices/${facturaId}${fechaCorte ? `?fecha_corte=${fechaCorte}` : ""}`),
  createInvoice: (payload: InvoiceCreateInput) =>
    request<Invoice>("/invoices", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  updateInvoice: (facturaId: string, payload: InvoiceUpdateInput) =>
    request<Invoice>(`/invoices/${facturaId}`, {
      method: "PATCH",
      body: JSON.stringify(payload)
    }),
  interactions: (facturaId: string, fechaCorte?: string) =>
    request<Interaction[]>(
      `/invoices/${facturaId}/interactions${fechaCorte ? `?fecha_corte=${fechaCorte}` : ""}`
    ),
  createInteraction: (payload: InteractionCreateInput) =>
    request<InteractionCreated>("/collections/interactions", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  createPromise: (payload: PromiseCreateInput) =>
    request<PromiseCreated>("/payment-promises", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  invoicePromises: (facturaId: string, fechaCorte?: string) =>
    request<PaymentPromise[]>(
      `/invoices/${facturaId}/promises${fechaCorte ? `?fecha_corte=${fechaCorte}` : ""}`
    ),
  updatePromise: (promesaId: string, payload: PromiseUpdateInput) =>
    request<PaymentPromise>(`/payment-promises/${promesaId}`, {
      method: "PATCH",
      body: JSON.stringify(payload)
    }),
  registerPayment: (payload: PaymentCreateInput) =>
    request<Invoice>("/payments", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  predictionHistory: (facturaId: string, fechaCorte?: string) =>
    request<PredictionHistoryItem[]>(
      `/invoices/${facturaId}/predictions${fechaCorte ? `?fecha_corte=${fechaCorte}` : ""}`
    ),
  predictionDaily: (facturaId: string, fechaCorte: string) =>
    request<PredictionDailyItem[]>(`/invoices/${facturaId}/prediction-daily?fecha_corte=${fechaCorte}`),
  customers: (limit = 100, offset = 0, query?: string) => {
    const params = new URLSearchParams({
      limit: String(limit),
      offset: String(offset)
    });
    const trimmedQuery = query?.trim();
    if (trimmedQuery) params.set("q", trimmedQuery);
    return request<Customer[]>(`/customers?${params.toString()}`);
  },
  customer: (clienteId: string) => request<Customer>(`/customers/${clienteId}`),
  segment: (clienteId: string) => request<Segment>(`/customers/${clienteId}/segment`),
  score: (facturaId: string, fechaCorte: string, persist = true, usePreparedSnapshot = false) =>
    request<Prediction>(`/invoices/${facturaId}/score`, {
      method: "POST",
      body: JSON.stringify({ fecha_corte: fechaCorte, persist, use_prepared_snapshot: usePreparedSnapshot })
    })
};
