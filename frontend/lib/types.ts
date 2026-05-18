export type DashboardSummary = {
  total_facturas: number;
  facturas_activas: number;
  monto_pendiente: number;
  monto_vencido: number;
  clientes_con_monto_vencido: number;
  promesas_activas: number;
  facturas_en_disputa: number;
};

export type PrioritizedInvoice = {
  factura_id: string;
  cliente_id: string;
  cliente_nombre: string;
  sector: string;
  monto: number;
  fecha_emision: string;
  fecha_vencimiento: string;
  fecha_pago_real: string | null;
  dias_mora_real: number | null;
  dias_mora_observable: number;
  estado_factura: string;
  estado_factura_actual: string;
  estado_corte: "preventive" | "overdue" | "paid";
  fecha_corte: string;
  predicted_label_usuario: string | null;
  prob_pago_plazo: number | null;
  prob_atraso_leve: number | null;
  prob_atraso_alto: number | null;
  prob_atraso_critico: number | null;
  any_late_probability: number | null;
  high_risk_probability: number | null;
  priority_score_0_100: number | null;
  accion_sugerida: string | null;
  rating_estrellas: number | null;
};

export type Invoice = {
  factura_id: string;
  cliente_id: string;
  fecha_emision: string;
  fecha_vencimiento: string;
  fecha_pago_real: string | null;
  condicion_dias: number;
  monto: number;
  saldo_pendiente: number;
  estado_factura: string;
  target_mora_simulado: string | null;
  dias_mora_real: number | null;
};

export type Customer = {
  cliente_id: string;
  nombre: string;
  sector: string;
  antiguedad_meses: number;
  tiene_garantia: boolean;
};

export type Segment = {
  cliente_id: string;
  cluster: number;
  tipo_cliente: string;
  riesgo_0_100: number;
  rating_estrellas: number;
  rating_label: string;
  por_que_rating: string;
  por_que_cluster: string;
  sector_dominante_modal: string | null;
  n_facturas_total: number | null;
  n_cortes_total: number | null;
};

export type Interaction = {
  gestion_id: string;
  fecha_gestion: string;
  canal: string;
  canal_label: string;
  contacto_exitoso: boolean;
  resultado: string;
  resultado_label: string;
  motivo_no_pago: string | null;
  motivo_no_pago_label: string | null;
  interpretacion: string;
  dias_mora_en_gestion: number;
};

export type Prediction = {
  factura_id: string;
  cliente_id: string;
  fecha_corte: string;
  modelo_version: string;
  predicted_class_tecnica: string;
  predicted_label_usuario: string;
  prob_pago_plazo: number;
  prob_atraso_leve: number;
  prob_atraso_alto: number;
  prob_atraso_critico: number;
  any_late_probability: number;
  high_risk_probability: number;
  priority_score_0_100: number;
  accion_sugerida: {
    codigo: string;
    nombre: string;
    canal_recomendado: string;
    severidad: number;
    motivo: string;
    regla: string;
  };
  feature_source: string;
};

export type PredictionHistoryItem = {
  prediccion_id: number;
  factura_id: string;
  cliente_id: string;
  fecha_corte: string;
  modelo_version: string;
  predicted_class_tecnica: string;
  predicted_label_usuario: string;
  prob_pago_plazo: number;
  prob_atraso_leve: number;
  prob_atraso_alto: number;
  prob_atraso_critico: number;
  any_late_probability: number;
  high_risk_probability: number;
  priority_score_0_100: number;
  accion_sugerida_codigo: string;
  accion_sugerida_nombre: string;
  motivo_accion: string;
  fecha_pago_real: string | null;
  dias_mora_real: number | null;
  target_mora_simulado: string | null;
};

export type PredictionDailyItem = {
  factura_id: string;
  cliente_id: string;
  fecha_corte: string;
  predicted_label_usuario: string;
  prob_pago_plazo: number;
  prob_atraso_leve: number;
  prob_atraso_alto: number;
  prob_atraso_critico: number;
  any_late_probability: number;
  high_risk_probability: number;
  priority_score_0_100: number;
  accion_sugerida: {
    codigo: string;
    nombre: string;
    canal_recomendado: string;
    severidad: number;
    motivo: string;
    regla: string;
  };
  feature_source: string;
};

export type RecalculateResult = {
  fecha_corte: string;
  total_evaluadas: number;
  total_con_error: number;
  errores: Array<{ factura_id: string; error: string }>;
};

export type InvoiceCreateInput = {
  factura_id?: string;
  cliente_id: string;
  fecha_emision: string;
  fecha_vencimiento: string;
  fecha_pago_real?: string | null;
  monto: number;
  saldo_pendiente?: number | null;
  estado_factura: "abierta" | "pagada" | "en_disputa" | "anulada" | "castigada";
};

export type InvoiceUpdateInput = Partial<{
  fecha_emision: string;
  fecha_vencimiento: string;
  monto: number;
  saldo_pendiente: number;
  estado_factura: "abierta" | "pagada" | "en_disputa" | "anulada" | "castigada";
}>;

export type InteractionCreateInput = {
  factura_id: string;
  fecha_gestion: string;
  canal: "whatsapp" | "email" | "llamada" | "visita" | "carta_notarial";
  contacto_exitoso: boolean;
  resultado: string;
  motivo_no_pago?: string | null;
  observacion?: string | null;
  recalculate?: boolean;
};

export type InteractionCreated = {
  gestion_id: string;
  factura_id: string;
  cliente_id: string;
  fecha_gestion: string;
  canal: string;
  contacto_exitoso: boolean;
  resultado: string;
  motivo_no_pago: string | null;
  dias_mora_en_gestion: number;
  observacion: string | null;
};

export type PromiseCreateInput = {
  gestion_id: string;
  fecha_compromiso: string;
  recalculate?: boolean;
};

export type PromiseCreated = {
  promesa_id: string;
  gestion_id: string;
  factura_id: string;
  cliente_id: string;
  fecha_promesa: string;
  fecha_compromiso: string;
  se_cumplio: boolean;
  estado_promesa: string;
};

export type PaymentPromise = PromiseCreated;

export type PromiseUpdateInput = {
  estado_promesa: "activa" | "cumplida" | "incumplida" | "cancelada" | "reemplazada";
  se_cumplio?: boolean | null;
};

export type PaymentCreateInput = {
  factura_id: string;
  fecha_pago: string;
};

export type QueueSnapshotItem = {
  factura_id: string;
  priority_score_0_100: number | null;
  accion_sugerida: string | null;
  position: number;
};

export type SimulationChange = {
  factura_id: string;
  score_delta: number | null;
  accion_anterior: string | null;
  accion_nueva: string | null;
  posicion_delta: number | null;
};
