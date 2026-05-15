export type DashboardSummary = {
  total_facturas: number;
  facturas_activas: number;
  monto_pendiente: number;
  monto_vencido: number;
  promesas_activas: number;
  facturas_en_disputa: number;
};

export type PrioritizedInvoice = {
  factura_id: string;
  cliente_id: string;
  cliente_nombre: string;
  sector: string;
  monto: number;
  fecha_vencimiento: string;
  estado_factura: string;
  estado_factura_actual: string;
  fecha_corte: string;
  predicted_label_usuario: string;
  any_late_probability: number;
  high_risk_probability: number;
  priority_score_0_100: number;
  accion_sugerida: string;
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
  contacto_exitoso: boolean;
  resultado: string;
  motivo_no_pago: string | null;
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

export type RecalculateResult = {
  fecha_corte: string;
  total_evaluadas: number;
  total_con_error: number;
  errores: Array<{ factura_id: string; error: string }>;
};
