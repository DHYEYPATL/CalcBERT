import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
} from "react-native";
import React, { useEffect, useState } from "react";
import { SafeAreaView } from "react-native-safe-area-context";
import { useNavigation, useFocusEffect } from "@react-navigation/native";
import { PieChart, LineChart } from "react-native-gifted-charts";
import { Ionicons, MaterialIcons, Feather } from "@expo/vector-icons";

import {
  getSubscriptions,
  getTopMerchants,
  getCorrectionsHistory,
  getConfidenceTrend,
  getSpendByCategory,
  getAlerts,
  getTodaySplits,
} from "../../services/api";
import { DEFAULT_USER_ID } from "../../constants/user";

const DashboardScreen = () => {
  const navigation = useNavigation<any>();

  const [loading, setLoading] = useState(true);
  const [subscriptions, setSubscriptions] = useState<any[]>([]);
  const [splitData, setSplitData] = useState<any[]>([]);
  const [totalAmount, setTotalAmount] = useState(0);
  const [alerts, setAlerts] = useState<string[]>([]);
  const [topMerchants, setTopMerchants] = useState<string[]>([]);
  const [corrections, setCorrections] = useState<any[]>([]);
  const [confidenceTrend, setConfidenceTrend] = useState<number[]>([]);

  const userId = DEFAULT_USER_ID;

  const fetchDashboardData = async () => {
    try {
      setLoading(true);

      const [
        subsRes,
        spendRes,
        topMerchantsRes,
        correctionsRes,
        trendRes,
        alertsRes,
        todaySplitsRes,
      ] = await Promise.all([
        getSubscriptions(userId),
        getSpendByCategory(userId),
        getTopMerchants(userId, 3),
        getCorrectionsHistory(userId, 10),
        getConfidenceTrend(userId, 7),
        getAlerts(userId),
        getTodaySplits(userId),
      ]);

      if (subsRes.status === "ok" && subsRes.subscriptions) {
        setSubscriptions(
          subsRes.subscriptions.map((s: any) => ({
            name: s.merchant || s.name || "Unknown",
            amount: s.amount || 0,
            cycle:
              s.period === "monthly"
                ? "Monthly"
                : s.period === "weekly"
                  ? "Weekly"
                  : "Yearly",
          })),
        );
      }

      if (
        todaySplitsRes.status === "ok" &&
        todaySplitsRes.combined_splits?.length > 0
      ) {
        setSplitData(todaySplitsRes.combined_splits);
        setTotalAmount(todaySplitsRes.total_amount || 0);
      } else if (spendRes.status === "ok" && spendRes.split_data) {
        setSplitData(spendRes.split_data);
        setTotalAmount(spendRes.total_amount || 0);
      }

      if (topMerchantsRes.status === "ok") {
        setTopMerchants(
          topMerchantsRes.merchants?.map((m: any) => m.merchant) || [],
        );
      }

      if (correctionsRes.status === "ok") {
        setCorrections(correctionsRes.corrections || []);
      }

      if (trendRes.status === "ok") {
        setConfidenceTrend(trendRes.values || []);
      }

      if (alertsRes.status === "ok") {
        setAlerts(
          alertsRes.alerts?.map((a: any) => a.message || a.type).slice(0, 5) ||
            [],
        );
      }
    } catch (err) {
      console.error("Dashboard error:", err);
    } finally {
      setLoading(false);
    }
  };

  useFocusEffect(
    React.useCallback(() => {
      fetchDashboardData();
    }, []),
  );

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const PIE_COLORS = ["#F97316", "#22C55E", "#3B82F6", "#A855F7", "#EC4899"];

  const pieData = splitData.map((item: any, index: number) => ({
    value: item.amount,
    color: PIE_COLORS[index % PIE_COLORS.length],
    text: `${Math.round((item.amount / totalAmount) * 100)}%`,
    label: item.label,
    amount: item.amount,
  }));

  const lineData =
    confidenceTrend.length > 0
      ? confidenceTrend.map((v) => ({ value: v }))
      : [{ value: 0 }];

  if (loading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loader}>
          <ActivityIndicator size="large" color="#F97316" />
          <Text style={styles.loaderText}>Loading dashboard...</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView showsVerticalScrollIndicator={false}>
        {/* HEADER */}
        <View style={styles.header}>
          <Text style={styles.title}>Daily Overview</Text>
          <Text style={styles.subtitle}>Your activity today</Text>
        </View>

        {/* SUMMARY WIDGETS */}
        <View style={styles.widgetRow}>
          <View style={styles.widgetCard}>
            <Ionicons name="wallet-outline" size={22} color="#F97316" />
            <Text style={styles.widgetValue}>₹{totalAmount}</Text>
            <Text style={styles.widgetLabel}>Today Spend</Text>
          </View>

          <View style={styles.widgetCard}>
            <MaterialIcons name="autorenew" size={22} color="#22C55E" />
            <Text style={styles.widgetValue}>{subscriptions.length}</Text>
            <Text style={styles.widgetLabel}>Subscriptions</Text>
          </View>
        </View>

        <View style={styles.widgetRow}>
          <View style={styles.widgetCard}>
            <Ionicons name="warning-outline" size={22} color="#FB923C" />
            <Text style={styles.widgetValue}>{alerts.length}</Text>
            <Text style={styles.widgetLabel}>Alerts</Text>
          </View>

          <View style={styles.widgetCard}>
            <Feather name="trending-up" size={22} color="#3B82F6" />
            <Text style={styles.widgetValue}>
              {confidenceTrend.at(-1) ?? 0}
            </Text>
            <Text style={styles.widgetLabel}>Confidence</Text>
          </View>
        </View>

        {/* QUICK ACTIONS */}
        <View style={styles.quickActionRow}>
          <TouchableOpacity
            style={styles.quickAction}
            onPress={() =>
              navigation.navigate("QRScannerScreen", {
                source: "HOME_UPI",
              })
            }
          >
            <Ionicons name="qr-code-outline" size={24} color="#F97316" />
            <Text style={styles.quickText}>Pay UPI</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.quickAction}
            onPress={() => navigation.navigate("TransactionHistoryScreen")}
          >
            <Ionicons name="time-outline" size={24} color="#F97316" />
            <Text style={styles.quickText}>History</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.quickAction}
            onPress={() => navigation.navigate("SubscriptionsScreen")}
          >
            <Ionicons name="repeat-outline" size={24} color="#F97316" />
            <Text style={styles.quickText}>Subs</Text>
          </TouchableOpacity>

          {/* <TouchableOpacity
            style={styles.quickAction}
            onPress={() => navigation.navigate("AlertsScreen")}
          >
            <Ionicons name="notifications-outline" size={24} color="#F97316" />
            <Text style={styles.quickText}>Alerts</Text>
          </TouchableOpacity> */}
          <TouchableOpacity style={styles.quickAction}>
            <Ionicons name="git-compare-outline" size={24} color="#F97316" />
            <Text style={styles.quickText}>Split</Text>
          </TouchableOpacity>
        </View>

        {/* ALERTS */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Alerts & Warnings</Text>
          {alerts.length > 0 ? (
            alerts.map((alert, index) => (
              <View key={index} style={styles.alertRow}>
                <Ionicons name="alert-circle" size={18} color="#FB923C" />
                <Text style={styles.alertText}>{alert}</Text>
              </View>
            ))
          ) : (
            <Text style={styles.emptyText}>No alerts</Text>
          )}
        </View>

        {/* PIE CHART */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Today’s Spend</Text>

          <View style={styles.chartCenter}>
            {pieData.length > 0 ? (
              <PieChart
                data={pieData}
                donut
                radius={90}
                innerRadius={55}
                showText
                textColor="#FFFFFF"
                textSize={12}
                centerLabelComponent={() => (
                  <View style={{ alignItems: "center" }}>
                    <Text style={styles.centerLabel}>₹{totalAmount}</Text>
                    <Text style={{ color: "#9CA3AF", fontSize: 12 }}>
                      Total Spend
                    </Text>
                  </View>
                )}
              />
            ) : (
              <Text style={styles.emptyText}>No data</Text>
            )}
          </View>

          {/* LEGEND */}
          <View style={{ marginTop: 16 }}>
            {pieData.map((item, index) => (
              <View
                key={index}
                style={{
                  flexDirection: "row",
                  alignItems: "center",
                  marginBottom: 8,
                }}
              >
                <View
                  style={{
                    width: 10,
                    height: 10,
                    borderRadius: 5,
                    backgroundColor: item.color,
                    marginRight: 8,
                  }}
                />
                <Text style={{ color: "#D1D5DB", flex: 1 }}>{item.label}</Text>
                <Text style={{ color: "#FFFFFF", fontWeight: "600" }}>
                  ₹{item.amount} ({item.text})
                </Text>
              </View>
            ))}
          </View>
        </View>

        {/* LINE CHART */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Confidence Trend</Text>
          <Text style={{ color: "#9CA3AF", fontSize: 12, marginBottom: 8 }}>
            Last 7 days confidence score
          </Text>

          <LineChart
            data={lineData}
            thickness={3}
            color="#F97316"
            hideDataPoints={false}
            dataPointsColor="#F97316"
            dataPointsRadius={4}
            yAxisColor="#1F2933"
            xAxisColor="#1F2933"
            yAxisTextStyle={{ color: "#9CA3AF", fontSize: 10 }}
            noOfSections={4}
            isAnimated
            areaChart
            startFillColor="#F97316"
            endFillColor="#0B0B0B"
            startOpacity={0.25}
            endOpacity={0}
          />

          <Text
            style={{
              color: "#22C55E",
              marginTop: 8,
              textAlign: "right",
            }}
          >
            Current confidence: {confidenceTrend.at(-1) ?? 0}
          </Text>
        </View>

        {/* SUBSCRIPTIONS */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Subscriptions</Text>
          {subscriptions.length > 0 ? (
            subscriptions.map((sub, i) => (
              <View key={i} style={styles.subscriptionRow}>
                <View>
                  <Text style={styles.subName}>{sub.name}</Text>
                  <Text style={styles.subCategory}>{sub.cycle}</Text>
                </View>
                <Text style={styles.subAmount}>₹{sub.amount}</Text>
              </View>
            ))
          ) : (
            <Text style={styles.emptyText}>No subscriptions</Text>
          )}
        </View>

        {/* TOP MERCHANTS */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Top Merchants</Text>
          {topMerchants.map((m, i) => (
            <Text key={i} style={styles.listItem}>
              • {m}
            </Text>
          ))}
        </View>

        {/* CORRECTIONS */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Corrections History</Text>
          {corrections.slice(0, 5).map((c, i) => (
            <Text key={i} style={styles.listItem}>
              • {c.original_text?.slice(0, 30)} → {c.corrected_category}
            </Text>
          ))}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

export default DashboardScreen;

/* ================= STYLES ================= */

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#0B0B0B" },

  loader: { flex: 1, justifyContent: "center", alignItems: "center" },
  loaderText: { color: "#fff", marginTop: 12 },

  header: { padding: 16 },
  title: { color: "#fff", fontSize: 22, fontWeight: "700" },
  subtitle: { color: "#9CA3AF", marginTop: 4 },

  widgetRow: {
    flexDirection: "row",
    paddingHorizontal: 16,
    marginBottom: 12,
  },

  widgetCard: {
    flex: 1,
    backgroundColor: "#111827",
    marginHorizontal: 6,
    padding: 14,
    borderRadius: 14,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#1F2933",
  },

  widgetValue: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "700",
    marginTop: 6,
  },

  widgetLabel: {
    color: "#9CA3AF",
    fontSize: 12,
  },

  quickActionRow: {
    flexDirection: "row",
    paddingHorizontal: 16,
    marginBottom: 16,
  },

  quickAction: {
    flex: 1,
    backgroundColor: "#111827",
    paddingVertical: 12,
    marginHorizontal: 6,
    borderRadius: 14,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#1F2933",
  },

  quickText: {
    color: "#fff",
    fontSize: 12,
    marginTop: 6,
  },

  card: {
    backgroundColor: "#111827",
    marginHorizontal: 16,
    marginBottom: 16,
    padding: 16,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "#1F2933",
  },

  cardTitle: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "600",
    marginBottom: 12,
  },

  alertRow: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 8,
  },

  alertText: {
    color: "#fff",
    marginLeft: 8,
    flex: 1,
  },

  chartCenter: {
    alignItems: "center",
    justifyContent: "center",
  },

  centerLabel: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "700",
  },

  subscriptionRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: "#1F2933",
  },

  subName: { color: "#fff", fontSize: 15 },
  subCategory: { color: "#9CA3AF", fontSize: 12 },
  subAmount: { color: "#F97316", fontWeight: "600" },

  listItem: { color: "#D1D5DB", marginTop: 6 },
  emptyText: { color: "#9CA3AF" },
});
