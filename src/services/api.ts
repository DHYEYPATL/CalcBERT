/**
 * API Service for CalcBERT Backend
 * Handles all API calls to the backend server
 */

const BACKEND_URL = 'http://localhost:8000'; // Change to your backend URL

// ============================================================================
// Original CalcBERT - Transaction Categorization
// ============================================================================

export interface PredictRequest {
  text: string;
  meta?: Record<string, any>;
}

export interface PredictResponse {
  category: string;
  confidence: number;
  explanation: {
    rule_hits?: string[];
    top_tokens?: Array<{ token: string; score: number }>;
    rationale?: string;
  };
  model_used: string;
}

/**
 * Predict transaction category (Original CalcBERT feature)
 */
export async function predictTransaction(
  request: PredictRequest
): Promise<PredictResponse> {
  try {
    const response = await fetch(`${BACKEND_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error predicting transaction:', error);
    throw error;
  }
}

export const PaymentSplit = async (
  userId: string,
  transactionId: string,
  totalAmount: number,
  splits: { label: string; amount: number }[]
) => {
  const res = await fetch(`${BACKEND_URL}/transactions/${transactionId}/split`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_id: userId,
      total_amount: totalAmount,
      splits,
    }),
  });

  if (!res.ok) {
    throw new Error("Failed to save split");
  }

  return await res.json();
};


export const getTransactions = async (userId: string) => {
  try {
    const response = await fetch(
      `${BACKEND_URL}/transactions?user_id=${userId}`
    )

    if (!response.ok) {
      throw new Error("Failed to fetch transactions")
    }

    return await response.json()
  } catch (error) {
    console.error("getTransactions error:", error)
    return { transactions: [] }
  }
}


export const createTransaction = async (transaction: {
  user_id: string
  merchant: string
  amount: number
  upi_id: string
  note?: string
}) => {
  const res = await fetch('/transactions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(transaction),
  })

  return res.json()
}


// ============================================================================
// CalcBERT v2 - Prepay Risk & Spend Intelligence
// ============================================================================

export interface PrepayCheckRequest {
  merchant: string;
  amount: number;
  note: string;
  upi_id: string;
  user_role: string;
  user_id: string;
  date: string;
}

export interface PrepayCheckResponse {
  decision: 'allow' | 'warn' | 'block' | 'allow_with_note';
  options: string[];
  analysis: {
    final_category: string;
    final_confidence: number;
    risk_flags: {
      high_amount: boolean;
      unverified_merchant: boolean;
      low_quality_note: boolean;
      policy_violation: boolean;
    };
    explanation: {
      category_reason?: string;
      policy_reason?: string;
      suggestions?: string[];
    };
    subscription: {
      is_subscription: boolean;
      confidence: number;
    };
    similar_expenses: any[];
  };
  check_id?: number;
}

/**
 * Check prepay expense with backend API
 */
export async function checkPrepayExpense(
  request: PrepayCheckRequest
): Promise<PrepayCheckResponse> {
  try {
    const response = await fetch(`${BACKEND_URL}/prepay/check`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error checking prepay expense:', error);
    throw error;
  }
}

/**
 * Get prepay check history
 */
export async function getPrepayHistory(userId?: string, limit: number = 10) {
  try {
    const url = new URL(`${BACKEND_URL}/prepay/history`);
    if (userId) url.searchParams.append('user_id', userId);
    url.searchParams.append('limit', limit.toString());

    const response = await fetch(url.toString());
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error fetching prepay history:', error);
    throw error;
  }
}

/**
 * Get summary for today
 */
export async function getTodaySummary(userId?: string) {
  try {
    const url = new URL(`${BACKEND_URL}/summary/today`);
    if (userId) url.searchParams.append('user_id', userId);

    const response = await fetch(url.toString());
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error fetching today summary:', error);
    throw error;
  }
}

/**
 * Get subscriptions (both auto-detected and user-added)
 */
export async function getSubscriptions(userId?: string) {
  try {
    const url = new URL(`${BACKEND_URL}/summary/subscriptions`);
    if (userId) url.searchParams.append('user_id', userId);

    const response = await fetch(url.toString());
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error fetching subscriptions:', error);
    throw error;
  }
}

/**
 * Add a user subscription
 */
export async function addSubscription(
  userId: string,
  name: string,
  amount: number,
  period: 'monthly' | 'yearly' = 'monthly'
) {
  try {
    const url = new URL(`${BACKEND_URL}/summary/subscriptions`);
    
    const response = await fetch(url.toString(), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        user_id: userId,
        name: name,
        amount: amount,
        period: period,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error adding subscription:', error);
    throw error;
  }
}

/**
 * Delete a user subscription
 */
export async function deleteSubscription(subscriptionId: number, userId?: string) {
  try {
    let url = `${BACKEND_URL}/summary/subscriptions/${subscriptionId}`;
    if (userId) {
      const urlObj = new URL(url);
      urlObj.searchParams.append('user_id', userId);
      url = urlObj.toString();
    }

    const response = await fetch(url, {
      method: 'DELETE',
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error deleting subscription:', error);
    throw error;
  }
}

/**
 * Get top merchants from database
 */
export async function getTopMerchants(userId?: string, limit: number = 5) {
  try {
    const url = new URL(`${BACKEND_URL}/summary/top-merchants`);
    if (userId) url.searchParams.append('user_id', userId);
    url.searchParams.append('limit', limit.toString());

    const response = await fetch(url.toString());
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error fetching top merchants:', error);
    throw error;
  }
}

/**
 * Get corrections history from database
 */
export async function getCorrectionsHistory(userId?: string, limit: number = 10) {
  try {
    const url = new URL(`${BACKEND_URL}/summary/corrections`);
    if (userId) url.searchParams.append('user_id', userId);
    url.searchParams.append('limit', limit.toString());

    const response = await fetch(url.toString());
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error fetching corrections:', error);
    throw error;
  }
}

/**
 * Get confidence trend data for chart
 */
export async function getConfidenceTrend(userId?: string, days: number = 7) {
  try {
    const url = new URL(`${BACKEND_URL}/summary/confidence-trend`);
    if (userId) url.searchParams.append('user_id', userId);
    url.searchParams.append('days', days.toString());

    const response = await fetch(url.toString());
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error fetching confidence trend:', error);
    throw error;
  }
}

/**
 * Get spend by category for today (for pie chart)
 */
export async function getSpendByCategory(userId?: string) {
  try {
    const url = new URL(`${BACKEND_URL}/summary/spend-by-category`);
    if (userId) url.searchParams.append('user_id', userId);

    const response = await fetch(url.toString());
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error fetching spend by category:', error);
    throw error;
  }
}

/**
 * Save payment split to database
 */
export async function savePaymentSplit(
  userId: string,
  totalAmount: number,
  splits: Array<{ label: string; amount: number }>
) {
  try {
    const payload = {
      user_id: userId,
      total_amount: totalAmount,
      splits: splits,
    };
    console.log('Saving payment split:', payload);
    
    const response = await fetch(`${BACKEND_URL}/splits`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('API Error Response:', errorText);
      throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
    }

    const result = await response.json();
    console.log('Payment split saved successfully:', result);
    return result;
  } catch (error) {
    console.error('Error saving payment split:', error);
    throw error;
  }
}

/**
 * Get latest payment split from database
 */
export async function getLatestPaymentSplit(userId?: string) {
  try {
    const url = new URL(`${BACKEND_URL}/splits/latest`);
    if (userId) url.searchParams.append('user_id', userId);

    const response = await fetch(url.toString());
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error fetching latest split:', error);
    throw error;
  }
}

/**
 * Get all payment splits from today, combined
 */
export async function getTodaySplits(userId?: string) {
  try {
    const url = new URL(`${BACKEND_URL}/splits/today`);
    if (userId) url.searchParams.append('user_id', userId);

    const response = await fetch(url.toString());
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error fetching today splits:', error);
    throw error;
  }
}

/**
 * Get alerts from database
 */
export async function getAlerts(userId?: string) {
  try {
    const url = new URL(`${BACKEND_URL}/summary/alerts`);
    if (userId) url.searchParams.append('user_id', userId);

    const response = await fetch(url.toString());
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error fetching alerts:', error);
    throw error;
  }
}

/**
 * Submit feedback/correction for a prediction
 */
export interface FeedbackRequest {
  text: string;
  correct_label: string;
  user_id?: string;
}

export interface FeedbackResponse {
  status: string;
  id: number;
  message: string;
}

export async function submitFeedback(
  request: FeedbackRequest
): Promise<FeedbackResponse> {
  try {
    const response = await fetch(`${BACKEND_URL}/feedback`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error submitting feedback:', error);
    throw error;
  }
}

/**
 * Get available categories for feedback
 */
export async function getCategories() {
  try {
    const response = await fetch(`${BACKEND_URL}/categories`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error fetching categories:', error);
    throw error;
  }
}
