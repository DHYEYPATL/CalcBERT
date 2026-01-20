/**
 * API Service for CalcBERT Backend
 * Handles all API calls to the backend server
 */

const BACKEND_URL = 'https://pubic-decadently-verda.ngrok-free.dev'

// ============================================================================
// Original CalcBERT - Transaction Categorization
// ============================================================================

export interface PredictRequest {
  text: string
  meta?: Record<string, any>
}

export interface PredictResponse {
  category: string
  confidence: number
  explanation: {
    rule_hits?: string[]
    top_tokens?: Array<{ token: string; score: number }>
    rationale?: string
  }
  model_used: string
}

export async function predictTransaction(
  request: PredictRequest
): Promise<PredictResponse> {
  console.log('🔮 Predicting transaction:', request.text.substring(0, 50) + '...')
  const res = await fetch(`${BACKEND_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })

  if (!res.ok) {
    const err = await res.text()
    console.error('❌ Predict failed:', res.status, err)
    throw new Error(`Predict failed: ${res.status} - ${err}`)
  }

  const data = await res.json()
  console.log('✅ Prediction:', data.category, `(${data.confidence.toFixed(2)})` )
  return data
}

// ============================================================================
// Prepay Check
// ============================================================================

export interface PrepayCheckRequest {
  merchant: string
  amount: number
  note: string
  upi_id: string      // ✅ Backend expects upi_id
  user_role: string
  user_id: string     // ✅ Backend expects user_id
  date: string
}

export interface PrepayCheckResponse {
  decision: 'allow' | 'warn' | 'block' | 'allow_with_note'
  options: string[]
  analysis: any
  check_id?: number
}

export async function checkPrepayExpense(
  request: PrepayCheckRequest
): Promise<PrepayCheckResponse> {
  console.log('🛡️ Checking prepay:', request.merchant, request.amount)
  const res = await fetch(`${BACKEND_URL}/prepay/check`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })

  if (!res.ok) {
    const err = await res.text()
    console.error('❌ Prepay check failed:', res.status, err)
    throw new Error(`Prepay check failed ${res.status}: ${err}`)
  }

  const data = await res.json()
  console.log('✅ Prepay decision:', data.decision)
  return data
}

export async function getPrepayHistory(userId?: string, limit = 10) {
  console.log('📜 Loading prepay history...')
  const url = new URL(`${BACKEND_URL}/prepay/history`)
  if (userId) url.searchParams.append('user_id', userId)  // ✅ user_id
  url.searchParams.append('limit', String(limit))

  const res = await fetch(url.toString())
  if (!res.ok) {
    console.error('❌ History fetch failed:', res.status)
    throw new Error('Failed to load prepay history')
  }

  const data = await res.json()
  console.log('✅ Loaded', data.history?.length ?? data.count ?? 0, 'prepay history items')
  return data
}

// ============================================================================
// TRANSACTIONS (Critical for dashboard)
// ============================================================================

export async function getTransactions(userId?: string, range: "daily" | "weekly" | "monthly" = "monthly") {
  console.log('📊 Fetching transactions...', range)
  const url = new URL(`${BACKEND_URL}/splits/transactions`)
  if (userId) url.searchParams.append('user_id', userId)
  url.searchParams.append('range', range)

  const res = await fetch(url.toString())
  if (!res.ok) {
    console.error('❌ Transactions fetch failed:', res.status)
    throw new Error('Transactions fetch failed')
  }
  const data = await res.json()
  console.log('✅ Loaded', (data.transactions?.length || 0), 'transactions')
  return data.transactions || []
}

// ============================================================================
// Payment Splits (SINGLE SOURCE OF TRUTH)
// ============================================================================

export async function savePaymentSplit(
  userId: string,
  totalAmount: number,
  splits: Array<{ label: string; amount: number }>
) {
  const payload = {
    user_id: userId,
    total_amount: totalAmount,
    splits,
  }
  console.log('💰 Saving split:', payload.user_id, payload.total_amount, '→', splits.map(s => `${s.label}:${s.amount}`).join(', '))
  const res = await fetch(`${BACKEND_URL}/splits`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const err = await res.text()
    console.error('❌ Split save FAILED:', res.status, err)
    throw new Error(`Split save failed: ${res.status} - ${err}`)
  }
  const result = await res.json()
  console.log('✅ Split SAVED! ID:', result.split_id || result)
  return result
}

/** Save a split for a specific transaction (from History → Split). Calls POST /splits/transactions/:id/split */
export async function saveTransactionSplit(
  transactionId: number,
  userId: string,
  totalAmount: number,
  splits: Array<{ label: string; amount: number }>
) {
  const payload = { user_id: userId, total_amount: totalAmount, splits }
  console.log('💰 Saving transaction split:', transactionId, totalAmount, '→', splits.map(s => `${s.label}:${s.amount}`).join(', '))
  const res = await fetch(`${BACKEND_URL}/splits/transactions/${transactionId}/split`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const err = await res.text()
    console.error('❌ Transaction split save FAILED:', res.status, err)
    throw new Error(`Transaction split failed: ${res.status} - ${err}`)
  }
  const result = await res.json()
  console.log('✅ Transaction split SAVED!', result)
  return result
}

export async function getLatestPaymentSplit(userId?: string) {
  console.log('📋 Loading latest split...')
  const url = new URL(`${BACKEND_URL}/splits/latest`)
  if (userId) url.searchParams.append('user_id', userId)  // ✅ user_id

  const res = await fetch(url.toString())
  if (!res.ok) throw new Error('Failed to load latest split')

  const data = await res.json()
  console.log('✅ Latest split loaded')
  return data
}

export async function getTodaySplits(userId?: string) {
  console.log('📅 Loading today splits...')
  const url = new URL(`${BACKEND_URL}/splits/today`)
  if (userId) url.searchParams.append('user_id', userId)  // ✅ user_id

  const res = await fetch(url.toString())
  if (!res.ok) throw new Error('Failed to load today splits')

  const data = await res.json()
  console.log('✅ Today splits:', data.splits?.length || 0)
  return data
}

// ============================================================================
// Summary & Analytics
// ============================================================================

export async function getTodaySummary(userId?: string) {
  console.log('📊 Loading today summary...')
  const url = new URL(`${BACKEND_URL}/summary/today`)
  if (userId) url.searchParams.append('user_id', userId)

  const res = await fetch(url.toString())
  if (!res.ok) throw new Error('Summary fetch failed')
  return res.json()
}

export async function getSpendByCategory(userId?: string, range: "daily" | "weekly" | "monthly" = "weekly") {
  console.log('🧮 Loading spend by category...', range)
  const url = new URL(`${BACKEND_URL}/summary/spend-by-category`)
  if (userId) url.searchParams.append('user_id', userId)
  url.searchParams.append('range', range)

  const res = await fetch(url.toString())
  if (!res.ok) throw new Error('Spend category fetch failed')
  return res.json()
}

// ============================================================================
// Subscriptions
// ============================================================================

export async function getSubscriptions(userId?: string) {
  console.log('📡 Loading subscriptions...')
  const url = new URL(`${BACKEND_URL}/summary/subscriptions`)
  if (userId) url.searchParams.append('user_id', userId)  // ✅ user_id

  const res = await fetch(url.toString())
  if (!res.ok) throw new Error('Failed to load subscriptions')
  return res.json()
}

export async function addSubscription(
  userId: string,
  name: string,
  amount: number,
  period: 'monthly' | 'yearly'
) {
  console.log('➕ Adding subscription:', name, amount)
  const res = await fetch(`${BACKEND_URL}/summary/subscriptions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      user_id: userId,    // ✅ user_id
      name, 
      amount, 
      period 
    }),
  })

  if (!res.ok) {
    const err = await res.text()
    throw new Error(`Add subscription failed: ${err}`)
  }
  return res.json()
}

export async function deleteSubscription(subscriptionId: number, userId?: string) {
  console.log('🗑️ Deleting subscription:', subscriptionId)
  let url = `${BACKEND_URL}/summary/subscriptions/${subscriptionId}`
  if (userId) url += `?user_id=${userId}`  // ✅ user_id

  const res = await fetch(url, { method: 'DELETE' })
  if (!res.ok) throw new Error('Failed to delete subscription')
  return res.json()
}

// ============================================================================
// Feedback & Categories
// ============================================================================

export interface FeedbackRequest {
  text: string
  correct_label: string
  user_id?: string     // ✅ user_id
}

export async function submitFeedback(request: FeedbackRequest) {
  console.log('📝 Submitting feedback for:', request.text.substring(0, 30))
  const res = await fetch(`${BACKEND_URL}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })

  if (!res.ok) throw new Error('Feedback failed')
  return res.json()
}

export async function getCategories() {
  console.log('🏷️ Loading categories...')
  const res = await fetch(`${BACKEND_URL}/categories`)
  if (!res.ok) throw new Error('Failed to load categories')
  return res.json()
}

// ============================================================================
// Alerts & Merchants
// ============================================================================

export async function getAlerts(userId?: string) {
  console.log('🚨 Loading alerts...')
  const url = new URL(`${BACKEND_URL}/summary/alerts`)
  if (userId) url.searchParams.append('user_id', userId)

  const res = await fetch(url.toString())
  if (!res.ok) throw new Error('Alerts fetch failed')
  return res.json()
}

export async function getTopMerchants(userId?: string, limit = 5) {
  console.log('🏪 Loading top merchants...')
  const url = new URL(`${BACKEND_URL}/summary/top-merchants`)
  if (userId) url.searchParams.append('user_id', userId)
  url.searchParams.append('limit', String(limit))

  const res = await fetch(url.toString())
  if (!res.ok) throw new Error('Top merchants fetch failed')
  return res.json()
}

// ============================================================================
// Corrections History & Confidence Trend
// ============================================================================

export async function getCorrectionsHistory(userId?: string, limit = 10) {
  console.log('📝 Loading corrections history...')
  const url = new URL(`${BACKEND_URL}/summary/corrections`)
  if (userId) url.searchParams.append('user_id', userId)
  url.searchParams.append('limit', String(limit))

  const res = await fetch(url.toString())
  if (!res.ok) throw new Error('Failed to load corrections history')
  return res.json()
}

export async function getConfidenceTrend(userId?: string, days = 7) {
  console.log('📈 Loading confidence trend...')
  const url = new URL(`${BACKEND_URL}/summary/confidence-trend`)
  if (userId) url.searchParams.append('user_id', userId)
  url.searchParams.append('days', String(days))

  const res = await fetch(url.toString())
  if (!res.ok) throw new Error('Failed to load confidence trend')
  return res.json()
}
