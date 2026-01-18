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
import {
  getSubscriptions,
  getTodaySummary,
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

  // Fetch all dashboard data from API
  const fetchDashboardData = async () => {
    try {
      setLoading(true);

      // Fetch all data in parallel
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

      // Set subscriptions
      if (subsRes.status === "ok" && subsRes.subscriptions) {
        setSubscriptions(
          subsRes.subscriptions.map((s: any) => ({
            name: s.merchant || s.name || "Unknown",
            amount: s.amount || 0,
            cycle: s.period === "monthly" ? "Monthly" : s.period === "weekly" ? "Weekly" : "Monthly",
          }))
        );
      }

      // Set spend by category - combine today's splits with transaction data
      // Priority: Today's splits > Transaction data
      if (todaySplitsRes.status === "ok" && todaySplitsRes.combined_splits && todaySplitsRes.combined_splits.length > 0) {
        // Use combined splits from today
        setSplitData(todaySplitsRes.combined_splits || []);
        setTotalAmount(todaySplitsRes.total_amount || 0);
      } else if (spendRes.status === "ok" && spendRes.split_data) {
        // Fallback to transaction data
        setSplitData(spendRes.split_data || []);
        setTotalAmount(spendRes.total_amount || 0);
      }

      // Set top merchants
      if (topMerchantsRes.status === "ok" && topMerchantsRes.merchants) {
        setTopMerchants(
          topMerchantsRes.merchants.map((m: any) => m.merchant)
        );
      }

      // Set corrections
      if (correctionsRes.status === "ok" && correctionsRes.corrections) {
        setCorrections(correctionsRes.corrections || []);
      }

      // Set confidence trend
      if (trendRes.status === "ok" && trendRes.values) {
        setConfidenceTrend(trendRes.values || []);
      }

      // Set alerts
      if (alertsRes.status === "ok" && alertsRes.alerts) {
        setAlerts(
          alertsRes.alerts.map((a: any) => a.message || a.type).slice(0, 5)
        );
      }
    } catch (error) {
      console.error("Error fetching dashboard data:", error);
    } finally {
      setLoading(false);
    }
  };

  // Fetch data on mount and when screen is focused
  useFocusEffect(
    React.useCallback(() => {
      fetchDashboardData();
    }, [])
  );

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const PIE_COLORS = [
    "#F97316",
    "#22C55E",
    "#3B82F6",
    "#A855F7",
    "#EC4899",
    "#EAB308",
    "#14B8A6",
    "#F43F5E",
    "#6366F1",
    "#84CC16",
  ];

  const pieData = splitData.map((item: any, index: number) => ({
    value: item.amount,
    color: PIE_COLORS[index % PIE_COLORS.length],
    text: item.label,
  }));

  const lineData =
    confidenceTrend.length > 0
      ? confidenceTrend.map((v) => ({ value: v }))
      : [
          { value: 0 },
          { value: 0 },
          { value: 0 },
          { value: 0 },
          { value: 0 },
        ];

  if (loading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={{ flex: 1, justifyContent: "center", alignItems: "center" }}>
          <ActivityIndicator size="large" color="#F97316" />
          <Text style={{ color: "#FFFFFF", marginTop: 16 }}>Loading dashboard...</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView showsVerticalScrollIndicator={false}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>Daily Overview</Text>
          <Text style={styles.subtitle}>Your activity today</Text>
        </View>

        {/* Action Buttons */}
        <View style={styles.actionRow}>
          <TouchableOpacity
            style={styles.primaryAction}
            onPress={() => navigation.navigate("PaymentScreen")}
          >
            <Text style={styles.primaryText}>Pay via UPI</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.secondaryAction}
            onPress={() => navigation.navigate("SplitPaymentScreen")}
          >
            <Text style={styles.secondaryText}>Split Payment</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.actionRow}>
          <TouchableOpacity
            style={styles.primaryAction}
            onPress={() => navigation.navigate("SubscriptionsScreen")}
          >
            <Text style={styles.primaryText}>Subscriptions</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.secondaryAction}
            onPress={() => navigation.navigate("AlertsScreen")}
          >
            <Text style={styles.secondaryText}>Alerts</Text>
          </TouchableOpacity>
        </View>

        {/* Alerts */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Alerts & Warnings</Text>
          {alerts.map((alert, index) => (
            <View key={index} style={styles.alertRow}>
              <Text style={styles.alertIcon}>⚠</Text>
              <Text style={styles.alertText}>{alert}</Text>
            </View>
          ))}
        </View>

        {/* Pie Chart */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Today’s Spend</Text>

          <View style={styles.chartCenter}>
            {splitData.length > 0 ? (
              <PieChart
                data={pieData}
                donut
                radius={80}
                innerRadius={45}
                centerLabelComponent={() => (
                  <Text style={styles.centerLabel}>₹{totalAmount}</Text>
                )}
              />
            ) : (
              <Text style={{ color: "#9CA3AF" }}>
                No split data yet
              </Text>
            )}
          </View>

          {/* Legend */}
          <View style={styles.legendContainer}>
            {splitData.map((item: any, index: number) => (
              <View key={item.label} style={styles.legendItemRow}>
                <View
                  style={[
                    styles.legendDot,
                    {
                      backgroundColor:
                        PIE_COLORS[index % PIE_COLORS.length],
                    },
                  ]}
                />
                <Text style={styles.legendText}>{item.label}</Text>
                <Text style={styles.legendAmount}>₹{item.amount}</Text>
              </View>
            ))}
          </View>
        </View>

        {/* Line Chart */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Confidence Trend</Text>
          <LineChart
            data={lineData}
            thickness={3}
            color="#F97316"
            hideDataPoints
            hideRules
            isAnimated
            areaChart
            startFillColor="#F97316"
            endFillColor="#0B0B0B"
            startOpacity={0.3}
            endOpacity={0}
          />
        </View>

        {/* Subscriptions */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Subscriptions</Text>
          {subscriptions.length > 0 ? (
            subscriptions.map((sub: any, index: number) => (
              <View key={index} style={styles.subscriptionRow}>
                <View>
                  <Text style={styles.subName}>{sub.name}</Text>
                  <Text style={styles.subCategory}>
                    {sub.cycle}
                  </Text>
                </View>
                <Text style={styles.subAmount}>
                  ₹{sub.amount}
                  {sub.cycle === "Monthly" ? "/mo" : "/yr"}
                </Text>
              </View>
            ))
          ) : (
            <Text style={{ color: "#9CA3AF" }}>
              No subscriptions added
            </Text>
          )}
        </View>

        {/* Top Merchants from Database */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Top Merchants</Text>
          {topMerchants.length > 0 ? (
            topMerchants.map((merchant, index) => (
              <Text key={index} style={styles.listItem}>
                • {merchant}
              </Text>
            ))
          ) : (
            <Text style={{ color: "#9CA3AF" }}>No merchant data yet</Text>
          )}
        </View>

        {/* Corrections History from Database */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Corrections History</Text>
          {corrections.length > 0 ? (
            corrections.slice(0, 5).map((correction, index) => (
              <Text key={index} style={styles.listItem}>
                • {correction.original_text?.substring(0, 30)}... → {correction.corrected_category}
              </Text>
            ))
          ) : (
            <Text style={{ color: "#9CA3AF" }}>No corrections yet</Text>
          )}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

export default DashboardScreen;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#0B0B0B",
  },

  header: {
    padding: 16,
  },

  title: {
    color: "#FFFFFF",
    fontSize: 22,
    fontWeight: "700",
  },

  subtitle: {
    color: "#9CA3AF",
    fontSize: 14,
    marginTop: 4,
  },

  actionRow: {
    flexDirection: "row",
    paddingHorizontal: 16,
    marginBottom: 16,
  },

  primaryAction: {
    flex: 1,
    backgroundColor: "#F97316",
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: "center",
    marginRight: 8,
  },

  primaryText: {
    color: "#000",
    fontSize: 15,
    fontWeight: "600",
  },

  secondaryAction: {
    flex: 1,
    borderWidth: 1,
    borderColor: "#F97316",
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: "center",
    marginLeft: 8,
  },

  secondaryText: {
    color: "#F97316",
    fontSize: 15,
    fontWeight: "600",
  },

  card: {
    backgroundColor: "#111827",
    marginHorizontal: 16,
    marginBottom: 16,
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#1F2933",
  },

  cardTitle: {
    color: "#FFFFFF",
    fontSize: 16,
    fontWeight: "600",
    marginBottom: 12,
  },

  alertRow: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 8,
  },

  alertIcon: {
    color: "#FB923C",
    marginRight: 8,
    fontSize: 16,
  },

  alertText: {
    color: "#FFFFFF",
    fontSize: 14,
    flex: 1,
  },

  chartCenter: {
    alignItems: "center",
    justifyContent: "center",
  },

  centerLabel: {
    color: "#FFFFFF",
    fontSize: 16,
    fontWeight: "600",
  },

  legendRow: {
    flexDirection: "row",
    justifyContent: "space-around",
    marginTop: 10,
  },

  legendItem: {
    color: "#D1D5DB",
    fontSize: 13,
  },

  subscriptionRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: "#1F2933",
  },

  subName: {
    color: "#FFFFFF",
    fontSize: 15,
    fontWeight: "500",
  },

  subCategory: {
    color: "#9CA3AF",
    fontSize: 13,
    marginTop: 2,
  },

  subAmount: {
    color: "#F97316",
    fontSize: 15,
    fontWeight: "600",
  },

  listItem: {
    color: "#D1D5DB",
    fontSize: 14,
    marginTop: 6,
  },
  legendContainer: {
    marginTop: 12,
  },

  legendItemRow: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 8,
  },

  legendDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: 8,
  },

  legendText: {
    color: "#D1D5DB",
    fontSize: 14,
    flex: 1,
  },

  legendAmount: {
    color: "#FFFFFF",
    fontSize: 14,
    fontWeight: "600",
  },
});
