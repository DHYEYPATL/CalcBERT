import { useNavigation, useRoute } from "@react-navigation/native";
import React, { useEffect, useState } from "react";
import { createTransaction } from "../../services/api";
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
import {
  checkPrepayExpense,
  getCategories,
  PrepayCheckResponse,
  submitFeedback,
} from "../../services/api";

const PrepayResultScreen = () => {
  const route = useRoute<any>();
  const navigation = useNavigation<any>();

  const { merchantName, amount, note, upiId } = route.params || {};

  const [loading, setLoading] = useState(true);
  const [result, setResult] = useState<PrepayCheckResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showCorrectionModal, setShowCorrectionModal] = useState(false);
  const [categories, setCategories] = useState<string[]>([]);
  const [loadingCategories, setLoadingCategories] = useState(false);
  const [submittingFeedback, setSubmittingFeedback] = useState(false);

  useEffect(() => {
    const fetchPrepayCheck = async () => {
      if (!merchantName || !amount) {
        setError("Missing merchant or amount");
        setLoading(false);
        return;
      }

      try {
        setLoading(true);
        const response = await checkPrepayExpense({
          merchant: merchantName,
          amount: parseFloat(amount.toString()),
          note: note || "",
          upi_id: upiId || "",
          user_role: "employee",
          user_id: DEFAULT_USER_ID,
          date: new Date().toISOString().split("T")[0],
        });
        console.log(
          "Prepay check response:",
          JSON.stringify(response, null, 2),
        );
        setResult(response);
      } catch (err: any) {
        setError(err.message || "Failed to check prepay expense");
        console.error("Prepay check error:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchPrepayCheck();
  }, [merchantName, amount, note, upiId]);

  // Load categories when modal opens
  useEffect(() => {
    if (showCorrectionModal && categories.length === 0) {
      loadCategories();
    }
  }, [showCorrectionModal]);

  const loadCategories = async () => {
    try {
      setLoadingCategories(true);
      const response = await getCategories();
      if (response.status === "ok" && response.categories) {
        setCategories(response.categories);
      }
    } catch (err) {
      console.error("Error loading categories:", err);
      Alert.alert("Error", "Failed to load categories");
    } finally {
      setLoadingCategories(false);
    }
  };

  const handleCorrectCategory = async (correctCategory: string) => {
    if (!result) return;

    try {
      setSubmittingFeedback(true);

      // Build transaction text for feedback
      const transactionText = `${merchantName} ${note || ""}`.trim();

      await submitFeedback({
        text: transactionText,
        correct_label: correctCategory,
        user_id: DEFAULT_USER_ID,
      });

      Alert.alert(
        "Success",
        `Category corrected to "${correctCategory}". Thank you for the feedback!`,
        [
          {
            text: "OK",
            onPress: () => {
              setShowCorrectionModal(false);
              navigation.goBack();
            },
          },
        ],
      );
    } catch (err: any) {
      console.error("Error submitting feedback:", err);
      Alert.alert("Error", err.message || "Failed to submit correction");
    } finally {
      setSubmittingFeedback(false);
    }
  };

  const category = result?.analysis.final_category || "Unknown";
  const confidence = result
    ? Math.round(result.analysis.final_confidence * 100)
    : 0;

  const riskWarnings: string[] = [];

  if (result?.analysis.risk_flags.high_amount) {
    riskWarnings.push("High amount transaction");
  }
  if (result?.analysis.risk_flags.unverified_merchant) {
    riskWarnings.push("Unverified merchant detected");
  }
  if (result?.analysis.risk_flags.low_quality_note) {
    riskWarnings.push("Low quality note - please add more details");
  }
  if (result?.analysis.risk_flags.policy_violation) {
    riskWarnings.push("Policy violation detected");
  }

  // ✅ FIX: derived trust score (0–10)
  const trustScore = (() => {
    if (!result) return "N/A";

    let score = Math.round(result.analysis.final_confidence * 10);
    const flags = result.analysis.risk_flags;

    if (flags.high_amount) score -= 2;
    if (flags.unverified_merchant) score -= 3;
    if (flags.low_quality_note) score -= 1;
    if (flags.policy_violation) score -= 4;

    return Math.max(0, Math.min(10, score));
  })();

  if (loading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#F97316" />
          <Text style={styles.loadingText}>Analyzing transaction...</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (error) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.header}>
          <TouchableOpacity onPress={() => navigation.goBack()}>
            <Text style={styles.backText}>←</Text>
          </TouchableOpacity>
          <Text style={styles.title}>Pre-Payment Check</Text>
          <View style={{ width: 24 }} />
        </View>
        <View style={styles.card}>
          <Text style={styles.errorText}>Error: {error}</Text>
          <TouchableOpacity
            style={styles.primaryButton}
            onPress={() => navigation.goBack()}
          >
            <Text style={styles.primaryText}>Go Back</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Text style={styles.backText}>←</Text>
        </TouchableOpacity>
        <Text style={styles.title}>Pre-Payment Check</Text>
        <View style={{ width: 24 }} />
      </View>

      <View style={styles.card}>
        <Text style={styles.merchantName}>{merchantName}</Text>
        <Text style={styles.amountText}>₹ {amount}</Text>
        {note ? <Text style={styles.noteText}>“{note}”</Text> : null}
      </View>

      <View style={styles.card}>
        <Text style={styles.sectionLabel}>Category Prediction</Text>
        <Text style={styles.sectionValue}>{category}</Text>
        {/* Show correction button if confidence is low or category is Unknown */}
        {(confidence < 50 ||
          category === "Unknown" ||
          category === "unknown") && (
          <TouchableOpacity
            style={styles.correctButton}
            onPress={() => setShowCorrectionModal(true)}
          >
            <Text style={styles.correctButtonText}>
              ✏️ Correct Category ({confidence}% confidence is low)
            </Text>
          </TouchableOpacity>
        )}
      </View>

      <View style={styles.card}>
        <Text style={styles.sectionLabel}>Confidence Meter</Text>
        <View style={styles.meterBackground}>
          <View style={[styles.meterFill, { width: `${confidence}%` }]} />
        </View>
        <Text style={styles.confidenceText}>
          {confidence}% confidence this is a legitimate payment
        </Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.sectionLabel}>Risk Warnings</Text>
        {riskWarnings.map((risk, index) => (
          <View key={index} style={styles.warningRow}>
            <Text style={styles.warningIcon}>⚠</Text>
            <Text style={styles.warningText}>{risk}</Text>
          </View>
        ))}
      </View>

      <View style={styles.card}>
        <Text style={styles.sectionLabel}>Merchant Trust Status</Text>
        <Text style={styles.trustText}>Trust Score: {trustScore} / 10</Text>
      </View>

      <View style={styles.footer}>
        <TouchableOpacity
          style={styles.primaryButton}
          onPress={async () => {
            try {
              const tx = await createTransaction({
                user_id: DEFAULT_USER_ID,
                merchant: merchantName,
                amount: Number(amount),
                upi_id: upiId,
                note,
              });

              // Navigate to history after successful payment
              navigation.replace("TransactionHistoryScreen");
            } catch (e) {
              Alert.alert("Payment Failed", "Could not complete payment");
            }
          }}
        >
          <Text style={styles.primaryText}>Continue Anyway</Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.secondaryButton}>
          <Text style={styles.secondaryText}>Edit Note</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.cancelButton}
          onPress={() => navigation.goBack()}
        >
          <Text style={styles.cancelText}>Cancel</Text>
        </TouchableOpacity>
      </View>

      {/* Correction Modal */}
      <Modal
        visible={showCorrectionModal}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setShowCorrectionModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Correct Category</Text>
              <TouchableOpacity onPress={() => setShowCorrectionModal(false)}>
                <Text style={styles.modalClose}>✕</Text>
              </TouchableOpacity>
            </View>

            <Text style={styles.modalSubtitle}>
              Current prediction: {category} ({confidence}% confidence)
            </Text>
            <Text style={styles.modalDescription}>
              Please select the correct category:
            </Text>

            {loadingCategories ? (
              <View style={styles.loadingContainer}>
                <ActivityIndicator size="large" color="#F97316" />
                <Text style={styles.loadingText}>Loading categories...</Text>
              </View>
            ) : (
              <FlatList
                data={categories}
                keyExtractor={(item) => item}
                renderItem={({ item }) => (
                  <TouchableOpacity
                    style={[
                      styles.categoryItem,
                      item === category && styles.categoryItemSelected,
                    ]}
                    onPress={() => handleCorrectCategory(item)}
                    disabled={submittingFeedback}
                  >
                    <Text
                      style={[
                        styles.categoryItemText,
                        item === category && styles.categoryItemTextSelected,
                      ]}
                    >
                      {item}
                      {item === category && " (current)"}
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
                <Text style={styles.loadingText}>Submitting feedback...</Text>
              </View>
            )}
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
};

export default PrepayResultScreen;

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

  backText: {
    color: "#F97316",
    fontSize: 22,
  },

  title: {
    color: "#FFFFFF",
    fontSize: 18,
    fontWeight: "600",
  },

  card: {
    backgroundColor: "#111827",
    marginHorizontal: 16,
    marginTop: 16,
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#1F2933",
  },

  merchantName: {
    color: "#FFFFFF",
    fontSize: 18,
    fontWeight: "600",
  },

  amountText: {
    color: "#F97316",
    fontSize: 22,
    marginTop: 6,
    fontWeight: "600",
  },

  noteText: {
    color: "#9CA3AF",
    marginTop: 6,
    fontSize: 14,
  },

  sectionLabel: {
    color: "#9CA3AF",
    fontSize: 13,
  },

  sectionValue: {
    color: "#FFFFFF",
    fontSize: 16,
    marginTop: 4,
  },

  meterBackground: {
    height: 8,
    backgroundColor: "#1F2933",
    borderRadius: 6,
    marginTop: 10,
  },

  meterFill: {
    height: 8,
    backgroundColor: "#F97316",
    borderRadius: 6,
  },

  confidenceText: {
    color: "#9CA3AF",
    fontSize: 13,
    marginTop: 8,
  },

  warningRow: {
    flexDirection: "row",
    alignItems: "center",
    marginTop: 8,
  },

  warningIcon: {
    color: "#FB923C",
    marginRight: 8,
  },

  warningText: {
    color: "#FFFFFF",
    fontSize: 14,
    flex: 1,
  },

  trustText: {
    color: "#22C55E",
    fontSize: 16,
    marginTop: 6,
    fontWeight: "600",
  },

  footer: {
    marginTop: "auto",
    padding: 16,
  },

  primaryButton: {
    backgroundColor: "#F97316",
    paddingVertical: 16,
    borderRadius: 14,
    alignItems: "center",
    marginBottom: 12,
  },

  primaryText: {
    color: "#000",
    fontSize: 16,
    fontWeight: "600",
  },

  secondaryButton: {
    borderWidth: 1,
    borderColor: "#F97316",
    paddingVertical: 14,
    borderRadius: 14,
    alignItems: "center",
    marginBottom: 12,
  },

  secondaryText: {
    color: "#F97316",
    fontSize: 15,
    fontWeight: "500",
  },

  cancelButton: {
    alignItems: "center",
    paddingVertical: 8,
  },

  cancelText: {
    color: "#9CA3AF",
    fontSize: 14,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  loadingText: {
    color: "#FFFFFF",
    marginTop: 16,
    fontSize: 16,
  },
  errorText: {
    color: "#F43F5E",
    fontSize: 16,
    marginBottom: 16,
  },
  correctButton: {
    marginTop: 12,
    paddingVertical: 10,
    paddingHorizontal: 12,
    backgroundColor: "#1F2933",
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "#F97316",
  },
  correctButtonText: {
    color: "#F97316",
    fontSize: 14,
    fontWeight: "500",
    textAlign: "center",
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: "rgba(0, 0, 0, 0.7)",
    justifyContent: "flex-end",
  },
  modalContent: {
    backgroundColor: "#111827",
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    maxHeight: "80%",
    paddingBottom: 32,
  },
  modalHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: "#1F2933",
  },
  modalTitle: {
    color: "#FFFFFF",
    fontSize: 20,
    fontWeight: "600",
  },
  modalClose: {
    color: "#9CA3AF",
    fontSize: 24,
    fontWeight: "300",
  },
  modalSubtitle: {
    color: "#9CA3AF",
    fontSize: 14,
    paddingHorizontal: 20,
    marginTop: 8,
  },
  modalDescription: {
    color: "#FFFFFF",
    fontSize: 16,
    paddingHorizontal: 20,
    marginTop: 12,
    marginBottom: 16,
  },
  categoryItem: {
    paddingVertical: 16,
    paddingHorizontal: 20,
    borderBottomWidth: 1,
    borderBottomColor: "#1F2933",
  },
  categoryItemSelected: {
    backgroundColor: "#1F2933",
  },
  categoryItemText: {
    color: "#FFFFFF",
    fontSize: 16,
  },
  categoryItemTextSelected: {
    color: "#F97316",
    fontWeight: "600",
  },
  emptyText: {
    color: "#9CA3AF",
    textAlign: "center",
    padding: 20,
    fontSize: 14,
  },
  loadingOverlay: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: "rgba(0, 0, 0, 0.7)",
    justifyContent: "center",
    alignItems: "center",
  },
});
