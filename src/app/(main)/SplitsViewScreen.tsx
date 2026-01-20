import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  ActivityIndicator,
} from "react-native";
import React, { useEffect, useState } from "react";
import { SafeAreaView } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";
import { useNavigation, useRoute, useFocusEffect } from "@react-navigation/native";
import { DEFAULT_USER_ID } from "../../constants/user";
import { getTodaySplits, getTransactions } from "../../services/api";

type SplitItem = {
  id: number;
  total_amount: number;
  splits: Array<{ label: string; amount: number }>;
  created_at: number;
  merchant?: string;
  date?: string;
};

const SplitsViewScreen = () => {
  const navigation = useNavigation<any>();
  const route = useRoute<any>();
  const userId = DEFAULT_USER_ID;
  const transaction = route.params?.transaction;

  const [splits, setSplits] = useState<SplitItem[]>([]);
  const [loading, setLoading] = useState(true);

  const loadSplits = async () => {
    try {
      setLoading(true);
      const response = await getTodaySplits(userId);
      if (response.status === "ok" && response.splits) {
        // If viewing a specific transaction's splits, filter for that
        if (transaction) {
          const matching = response.splits.filter((s: any) => 
            Math.abs(s.total_amount - transaction.amount) < 0.01
          );
          // Add transaction info to splits
          const splitsWithInfo = matching.map((split: any) => ({
            ...split,
            merchant: transaction.merchant,
            date: transaction.date || transaction.created_at,
          }));
          setSplits(splitsWithInfo);
        } else {
          // Get all splits and match with transactions
          const transactionsRes = await getTransactions(userId, "monthly");
          const transactions = Array.isArray(transactionsRes) ? transactionsRes : (transactionsRes?.transactions || []);
          
          const splitsWithTransactions = response.splits.map((split: any) => {
            const splitDate = new Date(split.created_at * 1000);
            const matchingTransaction = transactions.find((t: any) => {
              const transDate = new Date(t.created_at);
              return Math.abs(split.total_amount - t.amount) < 0.01 &&
                     Math.abs(splitDate.getTime() - transDate.getTime()) < 24 * 60 * 60 * 1000;
            });
            
            return {
              ...split,
              merchant: matchingTransaction?.merchant || "Unknown",
              date: matchingTransaction?.date || matchingTransaction?.created_at,
            };
          });
          
          setSplits(splitsWithTransactions);
        }
      }
    } catch (error) {
      console.error("Error loading splits:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadSplits();
  }, []);

  useFocusEffect(
    React.useCallback(() => {
      loadSplits();
    }, [])
  );

  const formatDate = (timestamp: number | string) => {
    if (typeof timestamp === 'string') {
      return new Date(timestamp).toLocaleDateString("en-IN", {
        day: "2-digit",
        month: "short",
        year: "numeric",
      });
    }
    const date = new Date(timestamp * 1000);
    return date.toLocaleDateString("en-IN", {
      day: "2-digit",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const renderSplit = ({ item }: { item: SplitItem }) => {
    const percentage = (amount: number) => ((amount / item.total_amount) * 100).toFixed(1);
    
    return (
      <View style={styles.splitCard}>
        <View style={styles.splitHeader}>
          <View style={{ flex: 1 }}>
            <Text style={styles.merchant}>{item.merchant || "Split Payment"}</Text>
            <Text style={styles.date}>
              {item.date ? formatDate(item.date) : formatDate(item.created_at)}
            </Text>
          </View>
          <View style={styles.totalAmountContainer}>
            <Text style={styles.totalAmount}>₹{item.total_amount.toFixed(2)}</Text>
            <Text style={styles.totalLabel}>Total</Text>
          </View>
        </View>

        <View style={styles.divider} />

        <View style={styles.splitsList}>
          <Text style={styles.splitsTitle}>Split Breakdown</Text>
          {item.splits.map((split, index) => (
            <View key={index} style={styles.splitRow}>
              <View style={styles.splitInfo}>
                <View style={styles.splitLabelRow}>
                  <View style={[styles.categoryDot, { backgroundColor: getCategoryColor(split.label) }]} />
                  <Text style={styles.splitLabel}>{split.label}</Text>
                </View>
                <Text style={styles.splitPercentage}>{percentage(split.amount)}%</Text>
              </View>
              <Text style={styles.splitAmount}>₹{split.amount.toFixed(2)}</Text>
            </View>
          ))}
        </View>

        <View style={styles.summaryRow}>
          <Text style={styles.summaryLabel}>Total Split:</Text>
          <Text style={styles.summaryAmount}>
            ₹{item.splits.reduce((sum, s) => sum + s.amount, 0).toFixed(2)}
          </Text>
        </View>
      </View>
    );
  };

  const getCategoryColor = (label: string): string => {
    const colors: { [key: string]: string } = {
      "Food": "#F97316",
      "Transport": "#3B82F6",
      "Shopping": "#A855F7",
      "Entertainment": "#EC4899",
      "Bills": "#22C55E",
      "Utilities": "#10B981",
      "Business": "#6366F1",
      "Groceries": "#F59E0B",
    };
    return colors[label] || "#6B7280";
  };

  return (
    <SafeAreaView style={styles.container}>
      {/* HEADER */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Ionicons name="arrow-back" size={24} color="#F97316" />
        </TouchableOpacity>
        <Text style={styles.title}>
          {transaction ? "Payment Splits" : "All Splits"}
        </Text>
        <View style={{ width: 24 }} />
      </View>

      {loading ? (
        <View style={styles.loader}>
          <ActivityIndicator size="large" color="#F97316" />
          <Text style={styles.loaderText}>Loading splits...</Text>
        </View>
      ) : (
        <FlatList
          data={splits}
          keyExtractor={(item) => String(item.id)}
          renderItem={renderSplit}
          contentContainerStyle={{ padding: 16 }}
          showsVerticalScrollIndicator={false}
          ListEmptyComponent={
            <View style={styles.emptyContainer}>
              <Ionicons name="git-compare-outline" size={48} color="#6B7280" />
              <Text style={styles.emptyText}>No splits yet</Text>
              <Text style={styles.emptySubtext}>
                Create splits from transaction history
              </Text>
            </View>
          }
        />
      )}
    </SafeAreaView>
  );
};

export default SplitsViewScreen;

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
  splitCard: {
    backgroundColor: "#111827",
    borderRadius: 14,
    padding: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: "#1F2933",
  },
  splitHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    marginBottom: 12,
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
  totalAmountContainer: {
    alignItems: "flex-end",
  },
  totalAmount: {
    color: "#F97316",
    fontSize: 20,
    fontWeight: "700",
  },
  totalLabel: {
    color: "#9CA3AF",
    fontSize: 11,
    marginTop: 2,
  },
  divider: {
    height: 1,
    backgroundColor: "#1F2933",
    marginVertical: 12,
  },
  splitsList: {
    marginBottom: 12,
  },
  splitsTitle: {
    color: "#9CA3AF",
    fontSize: 12,
    fontWeight: "600",
    marginBottom: 12,
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },
  splitRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 10,
    paddingHorizontal: 12,
    backgroundColor: "#0B0B0B",
    borderRadius: 8,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: "#1F2933",
  },
  splitInfo: {
    flex: 1,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  splitLabelRow: {
    flexDirection: "row",
    alignItems: "center",
    flex: 1,
  },
  categoryDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 10,
  },
  splitLabel: {
    color: "#D1D5DB",
    fontSize: 15,
    fontWeight: "500",
  },
  splitPercentage: {
    color: "#6B7280",
    fontSize: 12,
    marginLeft: 8,
  },
  splitAmount: {
    color: "#22C55E",
    fontSize: 16,
    fontWeight: "600",
    marginLeft: 12,
  },
  summaryRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: "#1F2933",
  },
  summaryLabel: {
    color: "#9CA3AF",
    fontSize: 14,
    fontWeight: "600",
  },
  summaryAmount: {
    color: "#F97316",
    fontSize: 16,
    fontWeight: "700",
  },
  emptyContainer: {
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 60,
  },
  emptyText: {
    color: "#9CA3AF",
    fontSize: 16,
    fontWeight: "600",
    marginTop: 16,
  },
  emptySubtext: {
    color: "#6B7280",
    fontSize: 14,
    marginTop: 8,
  },
});
