import { Ionicons } from "@expo/vector-icons";
import Slider from "@react-native-community/slider";
import { useFocusEffect, useNavigation } from "@react-navigation/native";
import React, { useEffect, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  FlatList,
  Modal,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { DEFAULT_USER_ID } from "../../constants/user";
import { addAlertRule, deleteAlertRule, getAlertRules, getAlerts, updateAlertRule } from "../../services/api";

/* ================= TYPES ================= */

type SystemAlert = {
  id: string;
  message: string;
  severity: "low" | "medium" | "high";
  date?: string;
  type?: string;
};

type AlertRule = {
  id: string;
  category: string;
  limit: number;
  enabled: boolean;
};

const AlertsScreen = () => {
  const navigation = useNavigation<any>();
  const userId = DEFAULT_USER_ID;

  /* 🔹 System alerts from API */
  const [systemAlerts, setSystemAlerts] = useState<SystemAlert[]>([]);
  const [loading, setLoading] = useState(true);

  /* 🔹 User rules */
  const [rules, setRules] = useState<AlertRule[]>([]);
  const [loadingRules, setLoadingRules] = useState(false);

  /* 🔹 Modal state */
  const [showModal, setShowModal] = useState(false);
  const [category, setCategory] = useState("");
  const [limit, setLimit] = useState(2000);

  /* ================= LOAD ALERTS ================= */

  const loadAlerts = async () => {
    try {
      setLoading(true);
      const response = await getAlerts(userId);
      if (response.status === "ok" && response.alerts) {
        const alerts = response.alerts.map((a: any, index: number) => ({
          id: a.type || `alert_${index}`,
          message: a.message || a.type || "Alert",
          severity: (a.severity || "low").toLowerCase() as "low" | "medium" | "high",
          type: a.type,
        }));
        setSystemAlerts(alerts);
      }
    } catch (error) {
      console.error("Error loading alerts:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAlerts();
    loadRules();
  }, []);

  useFocusEffect(
    React.useCallback(() => {
      loadAlerts();
      loadRules();
    }, [])
  );

  const loadRules = async () => {
    try {
      setLoadingRules(true);
      const response = await getAlertRules(userId);
      if (response.status === "ok" && response.rules) {
        const loadedRules = response.rules.map((r: any) => ({
          id: String(r.id),
          category: r.category,
          limit: r.limit,
          enabled: r.enabled,
        }));
        setRules(loadedRules);
      }
    } catch (error) {
      console.error("Error loading alert rules:", error);
    } finally {
      setLoadingRules(false);
    }
  };

  /* ================= HELPERS ================= */

  const toggleRule = async (id: string) => {
    const rule = rules.find(r => r.id === id);
    if (!rule) return;

    try {
      const newEnabled = !rule.enabled;
      await updateAlertRule(Number(id), newEnabled, undefined, userId);
      setRules(prev =>
        prev.map(r =>
          r.id === id ? { ...r, enabled: newEnabled } : r
        )
      );
    } catch (error) {
      console.error("Error toggling rule:", error);
      Alert.alert("Error", "Failed to update alert rule");
    }
  };

  const addRule = async () => {
    if (!category.trim()) return;

    try {
      const response = await addAlertRule(userId, category.trim(), limit, true);
      if (response.status === "ok") {
        await loadRules(); // Reload rules from server
        setCategory("");
        setLimit(2000);
        setShowModal(false);
      }
    } catch (error: any) {
      console.error("Error adding rule:", error);
      Alert.alert("Error", error.message || "Failed to add alert rule");
    }
  };

  const removeRule = async (id: string) => {
    try {
      await deleteAlertRule(Number(id), userId);
      await loadRules(); // Reload rules from server
    } catch (error: any) {
      console.error("Error deleting rule:", error);
      Alert.alert("Error", error.message || "Failed to delete alert rule");
    }
  };

  /* ================= RENDERERS ================= */

  const renderSystemAlert = ({ item }: { item: SystemAlert }) => {
    const color =
      item.severity === "high"
        ? "#F43F5E"
        : item.severity === "medium"
        ? "#FB923C"
        : "#FACC15";

    return (
      <View style={styles.alertCard}>
        <Ionicons name="alert-circle" size={20} color={color} />
        <View style={{ flex: 1, marginLeft: 10 }}>
          <Text style={styles.alertText}>{item.message}</Text>
          {item.date && <Text style={styles.alertDate}>{item.date}</Text>}
        </View>
      </View>
    );
  };

  const renderRule = ({ item }: { item: AlertRule }) => (
    <View style={styles.ruleCard}>
      <View style={{ flex: 1 }}>
        <Text style={styles.ruleTitle}>
          {item.category} Spend Alert
        </Text>
        <Text style={styles.ruleDesc}>
          Alert if {item.category} spend exceeds ₹{item.limit}/day
        </Text>
      </View>

      <View style={styles.ruleActions}>
        <Switch
          value={item.enabled}
          onValueChange={() => toggleRule(item.id)}
          thumbColor={item.enabled ? "#F97316" : "#9CA3AF"}
          trackColor={{ false: "#1F2933", true: "#FDBA74" }}
        />
        <TouchableOpacity
          style={styles.deleteButton}
          onPress={() => removeRule(item.id)}
        >
          <Ionicons name="trash-outline" size={18} color="#FB7185" />
        </TouchableOpacity>
      </View>
    </View>
  );

  return (
    <SafeAreaView style={styles.container}>
      {/* HEADER */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Ionicons name="arrow-back" size={24} color="#F97316" />
        </TouchableOpacity>
        <Text style={styles.title}>Alerts</Text>
        <View style={{ width: 24 }} />
      </View>

      {loading ? (
        <View style={styles.loader}>
          <ActivityIndicator size="large" color="#F97316" />
          <Text style={styles.loaderText}>Loading alerts...</Text>
        </View>
      ) : (
        <FlatList
          ListHeaderComponent={
            <>
              {/* SYSTEM ALERTS */}
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Alerts & Warnings</Text>
                {systemAlerts.length > 0 ? (
                  systemAlerts.map(alert => (
                    <View key={alert.id}>
                      {renderSystemAlert({ item: alert })}
                    </View>
                  ))
                ) : (
                  <Text style={styles.emptyText}>No alerts</Text>
                )}
              </View>

              {/* USER RULES HEADER */}
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Alert Rules</Text>
              </View>
            </>
          }
        data={rules}
        keyExtractor={item => item.id}
        renderItem={renderRule}
        ListFooterComponent={
          <TouchableOpacity
            style={styles.addRuleCard}
            onPress={() => setShowModal(true)}
          >
            <Ionicons name="add-circle-outline" size={22} color="#F97316" />
            <Text style={styles.addRuleText}>Add Alert Rule</Text>
          </TouchableOpacity>
        }
          contentContainerStyle={{ paddingBottom: 40 }}
          showsVerticalScrollIndicator={false}
        />
      )}

      {/* ================= MODAL ================= */}
      <Modal
        visible={showModal}
        animationType="slide"
        transparent
        onRequestClose={() => setShowModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Create Spend Alert</Text>

            {/* CATEGORY INPUT */}
            <Text style={styles.modalLabel}>Category</Text>
            <TextInput
              style={styles.input}
              placeholder="e.g. Food"
              placeholderTextColor="#6B7280"
              value={category}
              onChangeText={setCategory}
            />

            {/* SLIDER */}
            <Text style={styles.modalLabel}>
              Limit: ₹{limit}
            </Text>
            <Slider
              minimumValue={100}
              maximumValue={10000}
              step={10}
              value={limit}
              onValueChange={setLimit}
              minimumTrackTintColor="#F97316"
              maximumTrackTintColor="#1F2933"
              thumbTintColor="#F97316"
            />

            {/* ACTIONS */}
            <View style={styles.modalActions}>
              <TouchableOpacity
                style={styles.cancelBtn}
                onPress={() => setShowModal(false)}
              >
                <Text style={styles.cancelText}>Cancel</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.saveBtn}
                onPress={addRule}
              >
                <Text style={styles.saveText}>Add Alert</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
};

export default AlertsScreen;

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

  section: {
    paddingHorizontal: 16,
    marginTop: 16,
  },

  sectionTitle: {
    color: "#9CA3AF",
    fontSize: 13,
    marginBottom: 12,
    fontWeight: "500",
  },

  alertCard: {
    flexDirection: "row",
    backgroundColor: "#111827",
    padding: 14,
    borderRadius: 14,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: "#1F2933",
  },

  alertText: {
    color: "#FFFFFF",
    fontSize: 14,
    fontWeight: "500",
  },

  alertDate: {
    color: "#9CA3AF",
    fontSize: 12,
    marginTop: 4,
  },

  ruleCard: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    backgroundColor: "#111827",
    marginHorizontal: 16,
    marginBottom: 12,
    padding: 14,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "#1F2933",
  },

  ruleTitle: {
    color: "#FFFFFF",
    fontSize: 15,
    fontWeight: "600",
  },

  ruleDesc: {
    color: "#9CA3AF",
    fontSize: 12,
    marginTop: 2,
  },

  addRuleCard: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#111827",
    marginHorizontal: 16,
    marginTop: 8,
    padding: 16,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "#F97316",
  },

  addRuleText: {
    color: "#F97316",
    fontSize: 15,
    fontWeight: "600",
    marginLeft: 8,
  },

  modalOverlay: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.7)",
    justifyContent: "flex-end",
  },

  modalContent: {
    backgroundColor: "#111827",
    padding: 20,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
  },

  modalTitle: {
    color: "#FFFFFF",
    fontSize: 18,
    fontWeight: "600",
    marginBottom: 16,
  },

  modalLabel: {
    color: "#9CA3AF",
    marginTop: 12,
    marginBottom: 6,
  },

  input: {
    backgroundColor: "#0B0B0B",
    borderRadius: 10,
    padding: 12,
    color: "#FFFFFF",
    borderWidth: 1,
    borderColor: "#1F2933",
  },

  modalActions: {
    flexDirection: "row",
    marginTop: 20,
  },

  cancelBtn: {
    flex: 1,
    paddingVertical: 14,
    alignItems: "center",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#1F2933",
    marginRight: 8,
  },

  cancelText: {
    color: "#9CA3AF",
    fontWeight: "600",
  },

  saveBtn: {
    flex: 1,
    backgroundColor: "#F97316",
    paddingVertical: 14,
    alignItems: "center",
    borderRadius: 12,
    marginLeft: 8,
  },

  saveText: {
    color: "#000",
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
    paddingVertical: 20,
  },
  ruleActions: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
  },
  deleteButton: {
    padding: 4,
  },
});
