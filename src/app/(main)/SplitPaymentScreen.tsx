import Slider from "@react-native-community/slider";
import { useNavigation, useRoute } from "@react-navigation/native";
import React, { useEffect, useState } from "react";
import {
  ActivityIndicator,
  FlatList,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { DEFAULT_USER_ID } from "../../constants/user";
import { getTodaySplits, saveTransactionSplit } from "../../services/api";

type Split = { id: string; label: string; amount: number };

const SplitPaymentScreen = () => {
  const navigation = useNavigation<any>();
  const route = useRoute<any>();
  const userId = DEFAULT_USER_ID;

  const transaction = route.params?.transaction;

  const [totalAmount, setTotalAmount] = useState(0);
  const [categoryInput, setCategoryInput] = useState("");
  const [splits, setSplits] = useState<Split[]>([]);
  const [saving, setSaving] = useState(false);

  /* 🔐 Guard + Prefill */
  useEffect(() => {
    if (!transaction) {
      navigation.goBack();
      return;
    }

    setTotalAmount(transaction.amount);

    // Check if this transaction already has splits
    checkExistingSplits();
  }, []);

  const checkExistingSplits = async () => {
    try {
      const response = await getTodaySplits(userId);
      if (response.status === "ok" && response.splits) {
        // Filter splits that might match this transaction
        const matching = response.splits.filter((s: any) => 
          Math.abs(s.total_amount - transaction.amount) < 0.01
        );
        
        if (matching.length > 0) {
          // Transaction already has splits - redirect to view screen
          navigation.replace("SplitsViewScreen", { transaction });
          return;
        }
      }

      // ✅ Prefill existing splits if present in transaction object
      if (transaction.splits?.length) {
        setSplits(
          transaction.splits.map((s: any) => ({
            id: Date.now().toString() + Math.random(),
            label: s.label,
            amount: s.amount,
          }))
        );
      }
    } catch (error) {
      console.error("Error checking existing splits:", error);
    }
  };

  const addCategory = () => {
    if (!categoryInput) return;

    const used = splits.reduce((s, i) => s + i.amount, 0);
    const remaining = totalAmount - used;
    if (remaining <= 0) return;

    setSplits([
      ...splits,
      {
        id: Date.now().toString(),
        label: categoryInput,
        amount: remaining,
      },
    ]);
    setCategoryInput("");
  };

  const updateSplit = (id: string, value: number) => {
    const rounded = Math.round(value);
    const updated = splits.map(s =>
      s.id === id ? { ...s, amount: rounded } : s
    );

    const used = updated.reduce((s, i) => s + i.amount, 0);
    if (used > totalAmount) return;

    setSplits(updated);
  };

  const usedTotal = splits.reduce((s, i) => s + i.amount, 0);

  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.title}>Split • {transaction.merchant}</Text>

      {/* TOTAL */}
      <View style={styles.totalInputCard}>
        <Text style={styles.label}>Total Amount</Text>
        <TextInput
          style={[styles.totalInput, { opacity: 0.6 }]}
          value={`₹${totalAmount}`}
          editable={false}
        />
      </View>

      {/* ADD CATEGORY */}
      <View style={styles.categoryInputRow}>
        <TextInput
          style={styles.categoryInput}
          placeholder="Enter category"
          placeholderTextColor="#6B7280"
          value={categoryInput}
          onChangeText={setCategoryInput}
          onSubmitEditing={addCategory}
        />
        <TouchableOpacity style={styles.addButton} onPress={addCategory}>
          <Text style={styles.addText}>Add</Text>
        </TouchableOpacity>
      </View>

      {/* SPLITS */}
      <FlatList
        data={splits}
        keyExtractor={i => i.id}
        renderItem={({ item }) => (
          <View style={styles.card}>
            <Text style={styles.cardLabel}>{item.label}</Text>
            <Text style={styles.amount}>₹{item.amount}</Text>
            <Slider
              minimumValue={0}
              maximumValue={totalAmount}
              value={item.amount}
              step={10}
              minimumTrackTintColor="#F97316"
              maximumTrackTintColor="#1F2933"
              thumbTintColor="#F97316"
              onValueChange={v => updateSplit(item.id, v)}
            />
          </View>
        )}
      />

      <Text
        style={[
          styles.totalSummary,
          usedTotal === totalAmount ? styles.ok : styles.warn,
        ]}
      >
        Total: ₹{usedTotal} / ₹{totalAmount}
      </Text>


      {/* SAVE */}
      <TouchableOpacity
        style={[
          styles.saveButton,
          (saving || usedTotal !== totalAmount) && styles.disabled,
        ]}
        disabled={saving || usedTotal !== totalAmount}
        onPress={async () => {
          try {
            setSaving(true);
            await saveTransactionSplit(
              Number(transaction.id),
              userId,
              totalAmount,
              splits.map(({ label, amount }) => ({ label, amount }))
            );
            navigation.replace("TransactionHistoryScreen");
          } catch {
            alert("Failed to save split");
          } finally {
            setSaving(false);
          }
        }}
      >
        {saving ? (
          <ActivityIndicator color="#000" />
        ) : (
          <Text style={styles.saveText}>Done</Text>
        )}
      </TouchableOpacity>
    </SafeAreaView>
  );
};

export default SplitPaymentScreen;


/* ================= STYLES ================= */

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0B0B0B',
    padding: 16,
  },

  title: {
    color: '#FFFFFF',
    fontSize: 20,
    fontWeight: '600',
    marginBottom: 16,
  },

  totalInputCard: {
    backgroundColor: '#111827',
    padding: 16,
    borderRadius: 12,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#1F2933',
  },

  label: {
    color: '#9CA3AF',
    marginBottom: 6,
  },

  totalInput: {
    color: '#FFFFFF',
    fontSize: 18,
  },

  categoryInputRow: {
    flexDirection: 'row',
    marginBottom: 16,
  },

  categoryInput: {
    flex: 1,
    backgroundColor: '#111827',
    borderRadius: 10,
    paddingHorizontal: 14,
    paddingVertical: 12,
    color: '#FFFFFF',
    borderWidth: 1,
    borderColor: '#1F2933',
    marginRight: 8,
  },

  addButton: {
    backgroundColor: '#F97316',
    borderRadius: 10,
    paddingHorizontal: 16,
    justifyContent: 'center',
  },

  addText: {
    color: '#000',
    fontWeight: '600',
  },

  card: {
    backgroundColor: '#111827',
    padding: 16,
    borderRadius: 12,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#1F2933',
  },

  cardLabel: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '500',
  },

  amount: {
    color: '#F97316',
    fontSize: 18,
    marginVertical: 6,
  },

  totalSummary: {
    textAlign: 'center',
    fontSize: 16,
    marginTop: 12,
    fontWeight: '600',
  },

  ok: { color: '#22C55E' },
  warn: { color: '#FB923C' },

  saveButton: {
    backgroundColor: '#F97316',
    paddingVertical: 14,
    borderRadius: 12,
    marginTop: 16,
    alignItems: 'center',
  },

  disabled: {
    backgroundColor: '#6B7280',
  },

  saveText: {
    color: '#000',
    fontWeight: '600',
  },
})
