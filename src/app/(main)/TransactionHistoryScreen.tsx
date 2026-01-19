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
import { useNavigation } from "@react-navigation/native";
import { DEFAULT_USER_ID } from "../../constants/user";
import { getTransactions } from "../../services/api";

/* 🔹 Transaction Type */
type Transaction = {
  id: string;
  merchant: string;
  amount: number;
  date: string;
  status: "NOT_SPLIT" | "SPLIT";
};

const TransactionHistoryScreen = () => {
  const navigation = useNavigation<any>();

  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [loading, setLoading] = useState(true);

  /* 🔹 Fetch real transaction history */
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        setLoading(true);
        const res = await getTransactions(DEFAULT_USER_ID);
        setTransactions(res.transactions || []);
      } catch (err) {
        console.error("Failed to fetch transactions", err);
      } finally {
        setLoading(false);
      }
    };

    fetchHistory();
  }, []);

  const renderTransaction = ({ item }: { item: Transaction }) => {
    const isSplit = item.status === "SPLIT";

    return (
      <TouchableOpacity
        style={styles.card}
        activeOpacity={0.8}
        onPress={() =>
          navigation.navigate("SplitPaymentScreen", {
            transaction: item,
          })
        }
      >
        <View style={styles.row}>
          <View>
            <Text style={styles.merchant}>{item.merchant}</Text>
            <Text style={styles.date}>{item.date}</Text>
          </View>

          <View style={{ alignItems: "flex-end" }}>
            <Text style={styles.amount}>₹{item.amount}</Text>

            <View
              style={[
                styles.statusBadge,
                { backgroundColor: isSplit ? "#064E3B" : "#1F2933" },
              ]}
            >
              <Text
                style={[
                  styles.statusText,
                  { color: isSplit ? "#22C55E" : "#9CA3AF" },
                ]}
              >
                {isSplit ? "Split" : "Not Split"}
              </Text>
            </View>
          </View>
        </View>
      </TouchableOpacity>
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

      {/* LOADING */}
      {loading ? (
        <View style={styles.loader}>
          <ActivityIndicator size="large" color="#F97316" />
          <Text style={styles.loaderText}>Loading transactions...</Text>
        </View>
      ) : (
        <FlatList
          data={transactions}
          keyExtractor={(item) => item.id}
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
});
