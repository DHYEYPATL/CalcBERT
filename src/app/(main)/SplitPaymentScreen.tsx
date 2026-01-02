import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  FlatList,
} from 'react-native'
import React, { useState } from 'react'
import Slider from '@react-native-community/slider'
import { SafeAreaView } from 'react-native-safe-area-context'
import { useNavigation } from '@react-navigation/native'


const SplitPaymentScreen = () => {
  const [totalAmount, setTotalAmount] = useState<number>(0)
  const [categoryInput, setCategoryInput] = useState('')
  const [splits, setSplits] = useState<
    { id: string; label: string; amount: number }[]
  >([])
  const navigation = useNavigation<any>()

  // 🔹 Add a new category
  const addCategory = () => {
    if (!categoryInput || totalAmount <= 0) return

    const remaining =
      totalAmount -
      splits.reduce((sum, s) => sum + s.amount, 0)

    const newSplit = {
      id: Date.now().toString(),
      label: categoryInput,
      amount: Math.max(remaining, 0),
    }

    setSplits([...splits, newSplit])
    setCategoryInput('')
  }

  // 🔹 Update slider & rebalance others
  const updateSplit = (id: string, value: number) => {
    const rounded = Math.round(value)

    let updated = splits.map(s =>
      s.id === id ? { ...s, amount: rounded } : s
    )

    let used = updated.reduce((sum, s) => sum + s.amount, 0)

    if (used > totalAmount) return

    setSplits(updated)
  }

  const usedTotal = splits.reduce(
    (sum, s) => sum + s.amount,
    0
  )

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Split Payment</Text>

      {/* Total Amount Input */}
      <View style={styles.totalInputCard}>
        <Text style={styles.label}>Total Amount</Text>
        <TextInput
          style={styles.totalInput}
          placeholder="Enter total amount"
          placeholderTextColor="#6B7280"
          keyboardType="numeric"
          value={totalAmount ? totalAmount.toString() : ''}
          onChangeText={text =>
            setTotalAmount(Number(text) || 0)
          }
        />
      </View>

      {/* Category Input */}
      <View style={styles.categoryInputRow}>
        <TextInput
          style={styles.categoryInput}
          placeholder="Enter category (e.g. Food)"
          placeholderTextColor="#6B7280"
          value={categoryInput}
          onChangeText={setCategoryInput}
          onSubmitEditing={addCategory}
        />

        <TouchableOpacity
          style={styles.addButton}
          onPress={addCategory}
        >
          <Text style={styles.addText}>Add</Text>
        </TouchableOpacity>
      </View>

      {/* Dynamic Category Cards */}
      <FlatList
        data={splits}
        keyExtractor={item => item.id}
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
              onValueChange={value =>
                updateSplit(item.id, value)
              }
            />
          </View>
        )}
      />

      {/* Total Summary */}
      <Text
        style={[
          styles.totalSummary,
          usedTotal === totalAmount
            ? styles.ok
            : styles.warn,
        ]}
      >
        Total: ₹{usedTotal} / ₹{totalAmount}
      </Text>
      <TouchableOpacity
  style={{
    backgroundColor: '#F97316',
    paddingVertical: 14,
    borderRadius: 12,
    marginTop: 16,
    alignItems: 'center',
  }}
  onPress={() => {
    navigation.navigate('index', {
      splitData: splits,
      totalAmount,
    })
  }}
>
  <Text style={{ color: '#000', fontWeight: '600' }}>
    Done
  </Text>
</TouchableOpacity>
    </View>
  )
}

export default SplitPaymentScreen
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
    borderBottomWidth: 2,
    borderBottomColor: '#F97316',
    paddingVertical: 4,
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

  ok: {
    color: '#22C55E',
  },

  warn: {
    color: '#FB923C',
  },
})
