import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  FlatList,
} from 'react-native'
import React, { useState } from 'react'
import { SafeAreaView } from 'react-native-safe-area-context'
import { useNavigation } from '@react-navigation/native'

const SubscriptionsScreen = () => {
  const navigation = useNavigation<any>()

  const [name, setName] = useState('')
  const [amount, setAmount] = useState('')
  const [cycle, setCycle] = useState<'Monthly' | 'Yearly'>(
    'Monthly'
  )

  const [subscriptions, setSubscriptions] = useState<
    {
      id: string
      name: string
      amount: number
      cycle: 'Monthly' | 'Yearly'
    }[]
  >([])

  const addSubscription = () => {
    if (!name || !amount) return

    setSubscriptions([
      ...subscriptions,
      {
        id: Date.now().toString(),
        name,
        amount: Number(amount),
        cycle,
      },
    ])

    setName('')
    setAmount('')
    setCycle('Monthly')
  }

  const removeSubscription = (id: string) => {
    setSubscriptions(subscriptions.filter(s => s.id !== id))
  }

  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.title}>Subscriptions</Text>

      {/* Input Card */}
      <View style={styles.card}>
        <TextInput
          style={styles.input}
          placeholder="Subscription name"
          placeholderTextColor="#6B7280"
          value={name}
          onChangeText={setName}
        />

        <TextInput
          style={styles.input}
          placeholder="Amount"
          placeholderTextColor="#6B7280"
          keyboardType="numeric"
          value={amount}
          onChangeText={setAmount}
        />

        {/* Billing Cycle */}
        <View style={styles.cycleRow}>
          {['Monthly', 'Yearly'].map(c => (
            <TouchableOpacity
              key={c}
              style={[
                styles.cycleButton,
                cycle === c && styles.cycleActive,
              ]}
              onPress={() =>
                setCycle(c as 'Monthly' | 'Yearly')
              }
            >
              <Text
                style={[
                  styles.cycleText,
                  cycle === c && styles.cycleTextActive,
                ]}
              >
                {c}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        <TouchableOpacity
          style={styles.addButton}
          onPress={addSubscription}
        >
          <Text style={styles.addText}>Add Subscription</Text>
        </TouchableOpacity>
      </View>

      {/* Subscriptions List */}
      <FlatList
        data={subscriptions}
        keyExtractor={item => item.id}
        renderItem={({ item }) => (
          <View style={styles.subscriptionRow}>
            <View>
              <Text style={styles.subName}>{item.name}</Text>
              <Text style={styles.subCycle}>
                {item.cycle}
              </Text>
            </View>

            <View style={styles.amountCol}>
              <Text style={styles.subAmount}>
                ₹{item.amount}
              </Text>
              <TouchableOpacity
                onPress={() =>
                  removeSubscription(item.id)
                }
              >
                <Text style={styles.remove}>Remove</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}
      />

      {/* Done */}
      <TouchableOpacity
        style={styles.doneButton}
        onPress={() =>
          navigation.navigate('index', {
            subscriptions,
          })
        }
      >
        <Text style={styles.doneText}>Done</Text>
      </TouchableOpacity>
    </SafeAreaView>
  )
}

export default SubscriptionsScreen
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

  card: {
    backgroundColor: '#111827',
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#1F2933',
    marginBottom: 16,
  },

  input: {
    backgroundColor: '#0B0B0B',
    color: '#FFFFFF',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderWidth: 1,
    borderColor: '#1F2933',
    marginBottom: 12,
  },

  cycleRow: {
    flexDirection: 'row',
    marginBottom: 12,
  },

  cycleButton: {
    flex: 1,
    borderWidth: 1,
    borderColor: '#1F2933',
    paddingVertical: 10,
    alignItems: 'center',
    marginRight: 8,
    borderRadius: 8,
  },

  cycleActive: {
    backgroundColor: '#F97316',
    borderColor: '#F97316',
  },

  cycleText: {
    color: '#9CA3AF',
  },

  cycleTextActive: {
    color: '#000',
    fontWeight: '600',
  },

  addButton: {
    backgroundColor: '#F97316',
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: 'center',
  },

  addText: {
    color: '#000',
    fontWeight: '600',
  },

  subscriptionRow: {
    backgroundColor: '#111827',
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#1F2933',
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 12,
  },

  subName: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '500',
  },

  subCycle: {
    color: '#9CA3AF',
    marginTop: 4,
  },

  amountCol: {
    alignItems: 'flex-end',
  },

  subAmount: {
    color: '#F97316',
    fontSize: 16,
    fontWeight: '600',
  },

  remove: {
    color: '#FB7185',
    marginTop: 6,
    fontSize: 13,
  },

  doneButton: {
    backgroundColor: '#22C55E',
    paddingVertical: 16,
    borderRadius: 14,
    alignItems: 'center',
    marginTop: 8,
  },

  doneText: {
    color: '#000',
    fontSize: 16,
    fontWeight: '600',
  },
})
