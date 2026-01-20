import { Ionicons } from "@expo/vector-icons";
import { useNavigation } from "@react-navigation/native";
import React, { useEffect, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  FlatList,
  Modal,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { DEFAULT_USER_ID } from "../../constants/user";
import { getCategories, getTransactions, submitFeedback } from "../../services/api";

/* 🔹 Transaction: API shape from /splits/transactions (prepay_checks). Pass full item as transaction for split. */
type Transaction = {
  id: number | string;
  merchant: string;
  amount: number;
  date?: string;
  created_at?: string;
  decision?: string;
  note?: string;
  analysis?: { 
    final_category?: string;
    final_confidence?: number;
  };
  status?: "NOT_SPLIT" | "SPLIT";
};

const TransactionHistoryScreen = () => {
  const navigation = useNavigation<any>();

  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [loading, setLoading] = useState(true);
  const [range, setRange] = useState<"daily" | "weekly" | "monthly">("monthly");
  const [showCorrectionModal, setShowCorrectionModal] = useState(false);
  const [selectedTransaction, setSelectedTransaction] = useState<Transaction | null>(null);
  const [categories, setCategories] = useState<string[]>([]);
  const [loadingCategories, setLoadingCategories] = useState(false);
  const [submittingFeedback, setSubmittingFeedback] = useState(false);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        setLoading(true);
        const res = await getTransactions(DEFAULT_USER_ID, range);
        const list = Array.isArray(res) ? res : (res?.transactions || []);
        setTransactions(list);
      } catch (err) {
        console.error("Failed to fetch transactions", err);
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
  }, [range]);

  useEffect(() => {
    if (showCorrectionModal && categories.length === 0) {
      loadCategories();
    }
  }, [showCorrectionModal]);

  const loadCategories = async () => {
    try {
      setLoadingCategories(true);
      const response = await getCategories();
      if (response.categories && Array.isArray(response.categories)) {
        setCategories(response.categories);
      }
    } catch (error) {
      console.error("Error loading categories:", error);
    } finally {
      setLoadingCategories(false);
    }
  };

  const handleCorrectCategory = async (correctCategory: string) => {
    if (!selectedTransaction) return;

    try {
      setSubmittingFeedback(true);
      const transactionText = `${selectedTransaction.merchant} ${selectedTransaction.note || ""}`.trim();

      await submitFeedback({
        text: transactionText,
        correct_label: correctCategory,
        user_id: DEFAULT_USER_ID,
      });

      Alert.alert(
        "Success",
        `Category corrected to "${correctCategory}". The model will learn from this correction for similar transactions.`,
        [
          {
            text: "OK",
            onPress: () => {
              setShowCorrectionModal(false);
              setSelectedTransaction(null);
              // Reload transactions to reflect any updates
              const fetchHistory = async () => {
                try {
                  const res = await getTransactions(DEFAULT_USER_ID, range);
                  const list = Array.isArray(res) ? res : (res?.transactions || []);
                  setTransactions(list);
                } catch (err) {
                  console.error("Failed to fetch transactions", err);
                }
              };
              fetchHistory();
            },
          },
        ]
      );
    } catch (err: any) {
      console.error("Error submitting feedback:", err);
      Alert.alert("Error", err.message || "Failed to submit correction");
    } finally {
      setSubmittingFeedback(false);
    }
  };

  const renderTransaction = ({ item }: { item: Transaction }) => {
    const isSplit = item.status === "SPLIT";
    const dateStr = item.date || (item.created_at ? new Date(item.created_at).toISOString().split("T")[0] : "—");
    const cat = item.analysis?.final_category;
    const confidence = item.analysis?.final_confidence || 0;
    const isLowConfidence = confidence < 0.7; // Threshold for low confidence

    return (
      <View style={styles.card}>
        <TouchableOpacity
          activeOpacity={0.8}
          onPress={() => {
            if (isSplit) {
              navigation.navigate("SplitsViewScreen", { transaction: item });
            } else {
              navigation.navigate("SplitPaymentScreen", { transaction: item });
            }
          }}
        >
          <View style={styles.row}>
            <View style={{ flex: 1 }}>
              <Text style={styles.merchant}>{item.merchant}</Text>
              <Text style={styles.date}>
                {dateStr}{cat ? ` • ${cat}` : ""}
                {isLowConfidence && (
                  <Text style={styles.lowConfidenceBadge}> • Low confidence ({Math.round(confidence * 100)}%)</Text>
                )}
              </Text>
            </View>
            <View style={{ alignItems: "flex-end" }}>
              <Text style={styles.amount}>₹{item.amount}</Text>
              <View style={[styles.statusBadge, { backgroundColor: isSplit ? "#064E3B" : "#1F2933" }]}>
                <Text style={[styles.statusText, { color: isSplit ? "#22C55E" : "#9CA3AF" }]}>
                  {isSplit ? "Split" : "Apply split"}
                </Text>
              </View>
            </View>
          </View>
        </TouchableOpacity>
        {isLowConfidence && (
          <TouchableOpacity
            style={styles.correctButton}
            onPress={() => {
              setSelectedTransaction(item);
              setShowCorrectionModal(true);
            }}
          >
            <Ionicons name="create-outline" size={16} color="#F97316" />
            <Text style={styles.correctButtonText}>Correct Category</Text>
          </TouchableOpacity>
        )}
      </View>
    );
  };

  return (
    <SafeAreaView style={styles.container}>
      {/* HEADER */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Ionicons name="arrow-back" size={24} color="#F97316" />
        </TouchableOpacity>

        <Text style={styles.title}>Transaction History</Text>

        <View style={{ width: 24 }} />
      </View>

      {/* DAILY / WEEKLY / MONTHLY */}
      <View style={styles.rangeRow}>
        {(["daily", "weekly", "monthly"] as const).map((r) => (
          <TouchableOpacity
            key={r}
            style={[styles.rangeTab, range === r && styles.rangeTabActive]}
            onPress={() => setRange(r)}
          >
            <Text style={[styles.rangeTabText, range === r && styles.rangeTabTextActive]}>{r.charAt(0).toUpperCase() + r.slice(1)}</Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* LOADING */}
      {loading ? (
        <View style={styles.loader}>
          <ActivityIndicator size="large" color="#F97316" />
          <Text style={styles.loaderText}>Loading transactions...</Text>
        </View>
      ) : (
        <FlatList
          data={transactions}
          keyExtractor={(item) => String(item.id)}
          renderItem={renderTransaction}
          contentContainerStyle={{ padding: 16 }}
          showsVerticalScrollIndicator={false}
          ListEmptyComponent={
            <Text style={styles.emptyText}>
              No transactions yet
            </Text>
          }
        />
      )}

      {/* Category Correction Modal */}
      <Modal
        visible={showCorrectionModal}
        animationType="slide"
        transparent
        onRequestClose={() => setShowCorrectionModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Correct Category</Text>
              <TouchableOpacity onPress={() => setShowCorrectionModal(false)}>
                <Ionicons name="close" size={24} color="#9CA3AF" />
              </TouchableOpacity>
            </View>
            {selectedTransaction && (
              <View style={styles.transactionInfo}>
                <Text style={styles.transactionMerchant}>{selectedTransaction.merchant}</Text>
                <Text style={styles.transactionAmount}>₹{selectedTransaction.amount}</Text>
                <Text style={styles.transactionCategory}>
                  Current: {selectedTransaction.analysis?.final_category || "Unknown"}
                </Text>
              </View>
            )}
            <Text style={styles.modalSubtitle}>Select the correct category:</Text>
            {loadingCategories ? (
              <View style={styles.loadingContainer}>
                <ActivityIndicator size="large" color="#F97316" />
                <Text style={styles.loaderText}>Loading categories...</Text>
              </View>
            ) : (
              <FlatList
                data={categories}
                keyExtractor={(item) => item}
                renderItem={({ item }) => (
                  <TouchableOpacity
                    style={[
                      styles.categoryItem,
                      item === selectedTransaction?.analysis?.final_category && styles.categoryItemSelected,
                    ]}
                    onPress={() => handleCorrectCategory(item)}
                    disabled={submittingFeedback}
                  >
                    <Text
                      style={[
                        styles.categoryItemText,
                        item === selectedTransaction?.analysis?.final_category && styles.categoryItemTextSelected,
                      ]}
                    >
                      {item}
                      {item === selectedTransaction?.analysis?.final_category && " (current)"}
                    </Text>
                  </TouchableOpacity>
                )}
                ListEmptyComponent={
                  <Text style={styles.emptyText}>No categories available</Text>
                }
              />
            )}
            {submittingFeedback && (
              <View style={styles.loadingOverlay}>
                <ActivityIndicator size="large" color="#F97316" />
                <Text style={styles.loaderText}>Submitting correction...</Text>
              </View>
            )}
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
};

export default TransactionHistoryScreen;

/* ================= STYLES ================= */

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#0B0B0B",
  },

  header: {
    height: 64,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 16,
    borderBottomWidth: 1,
    borderBottomColor: "#1F2933",
  },

  title: {
    color: "#FFFFFF",
    fontSize: 18,
    fontWeight: "600",
  },

  loader: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },

  loaderText: {
    color: "#FFFFFF",
    marginTop: 12,
    fontSize: 14,
  },

  emptyText: {
    color: "#9CA3AF",
    textAlign: "center",
    marginTop: 40,
    fontSize: 14,
  },

  card: {
    backgroundColor: "#111827",
    borderRadius: 14,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: "#1F2933",
  },

  row: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },

  merchant: {
    color: "#FFFFFF",
    fontSize: 16,
    fontWeight: "600",
  },

  date: {
    color: "#9CA3AF",
    fontSize: 12,
    marginTop: 4,
  },

  amount: {
    color: "#F97316",
    fontSize: 16,
    fontWeight: "600",
  },

  statusBadge: {
    marginTop: 6,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 10,
  },

  statusText: {
    fontSize: 12,
    fontWeight: "500",
  },

  rangeRow: { flexDirection: "row", paddingHorizontal: 16, paddingVertical: 12, gap: 8, borderBottomWidth: 1, borderBottomColor: "#1F2933" },
  rangeTab: { flex: 1, paddingVertical: 8, borderRadius: 8, alignItems: "center", backgroundColor: "#111827", borderWidth: 1, borderColor: "#1F2933" },
  rangeTabActive: { backgroundColor: "#F97316", borderColor: "#F97316" },
  rangeTabText: { color: "#9CA3AF", fontSize: 13, fontWeight: "600" },
  rangeTabTextActive: { color: "#000" },
  lowConfidenceBadge: {
    color: "#FB923C",
    fontSize: 11,
    fontWeight: "500",
  },
  correctButton: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#1F2933",
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 8,
    marginTop: 8,
    borderWidth: 1,
    borderColor: "#F97316",
  },
  correctButtonText: {
    color: "#F97316",
    fontSize: 13,
    fontWeight: "600",
    marginLeft: 6,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.7)",
    justifyContent: "flex-end",
  },
  modalContent: {
    backgroundColor: "#111827",
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    padding: 20,
    maxHeight: "80%",
  },
  modalHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 16,
  },
  modalTitle: {
    color: "#FFFFFF",
    fontSize: 18,
    fontWeight: "600",
  },
  transactionInfo: {
    backgroundColor: "#0B0B0B",
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: "#1F2933",
  },
  transactionMerchant: {
    color: "#FFFFFF",
    fontSize: 16,
    fontWeight: "600",
  },
  transactionAmount: {
    color: "#F97316",
    fontSize: 18,
    fontWeight: "700",
    marginTop: 4,
  },
  transactionCategory: {
    color: "#9CA3AF",
    fontSize: 12,
    marginTop: 4,
  },
  modalSubtitle: {
    color: "#9CA3AF",
    fontSize: 14,
    marginBottom: 12,
  },
  categoryItem: {
    backgroundColor: "#0B0B0B",
    padding: 14,
    borderRadius: 8,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: "#1F2933",
  },
  categoryItemSelected: {
    backgroundColor: "#1F2933",
    borderColor: "#F97316",
  },
  categoryItemText: {
    color: "#FFFFFF",
    fontSize: 14,
  },
  categoryItemTextSelected: {
    color: "#F97316",
    fontWeight: "600",
  },
  loadingContainer: {
    padding: 40,
    alignItems: "center",
  },
  loadingOverlay: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: "rgba(0,0,0,0.7)",
    justifyContent: "center",
    alignItems: "center",
    borderRadius: 20,
  },
});
