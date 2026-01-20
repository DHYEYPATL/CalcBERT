import { useFocusEffect, useNavigation } from '@react-navigation/native'
import React, { useEffect, useState } from 'react'
import {
  ActivityIndicator,
  Alert,
  FlatList,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
  Modal,
} from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'
import { DEFAULT_USER_ID } from '../../constants/user'
import { addSubscription, deleteSubscription, getSubscriptions } from '../../services/api'

const SubscriptionsScreen = () => {
  const navigation = useNavigation<any>()

  const userId = DEFAULT_USER_ID

  const [name, setName] = useState('')
  const [amount, setAmount] = useState('')
  const [cycle, setCycle] = useState<'Weekly' | 'Monthly' | 'Yearly'>(
    'Monthly'
  )

  const [subscriptions, setSubscriptions] = useState<
    {
      id: string
      name: string
      amount: number
      cycle: 'Weekly' | 'Monthly' | 'Yearly'
      db_id?: number
      source?: string
    }[]
  >([])
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [showDetectedModal, setShowDetectedModal] = useState(false)
  const [detectedSubscription, setDetectedSubscription] = useState<{
    name: string
    amount: number
    cycle: 'Weekly' | 'Monthly' | 'Yearly'
  } | null>(null)

  const [filterCycle, setFilterCycle] = useState<'All' | 'Weekly' | 'Monthly' | 'Yearly'>('All')

  // Load subscriptions from API
  const loadSubscriptions = async () => {
    try {
      setLoading(true)
      const response = await getSubscriptions(userId)
      if (response.status === 'ok' && response.subscriptions) {
        const subs = response.subscriptions.map((s: any) => {
          let cycle: 'Weekly' | 'Monthly' | 'Yearly' = 'Monthly'
          if (s.period === 'yearly') cycle = 'Yearly'
          else if (s.period === 'weekly') cycle = 'Weekly'
          else cycle = 'Monthly'
          
          return {
            id: s.id || s.name,
            name: s.name || s.merchant,
            amount: s.amount || 0,
            cycle,
            db_id: s.db_id,
            source: s.source,
          }
        })
        setSubscriptions(subs)
        
        // Check for newly detected subscriptions (not yet added by user)
        const detected = subs.find((s: any) => s.source === 'detected' && !s.db_id)
        if (detected) {
          // Check if we've already shown this prompt (using AsyncStorage or similar)
          // For now, show prompt for first detected subscription
          setDetectedSubscription({
            name: detected.name,
            amount: detected.amount,
            cycle: detected.cycle,
          })
          setShowDetectedModal(true)
        }
      }
    } catch (error: any) {
      console.error('Error loading subscriptions:', error)
      Alert.alert('Error', `Failed to load subscriptions: ${error?.message || 'Unknown error'}`)
    } finally {
      setLoading(false)
    }
  }

  // Load on mount and when screen is focused
  useEffect(() => {
    loadSubscriptions()
  }, [])

  useFocusEffect(
    React.useCallback(() => {
      loadSubscriptions()
    }, [])
  )

  const handleAddSubscription = async () => {
    if (!name || !amount) {
      Alert.alert('Validation', 'Please enter both name and amount')
      return
    }

    try {
      setSaving(true)
      let period: 'monthly' | 'yearly' | 'weekly' = 'monthly'
      if (cycle === 'Yearly') period = 'yearly'
      else if (cycle === 'Weekly') period = 'weekly'
      else period = 'monthly'
      
      await addSubscription(userId, name, Number(amount), period)
      Alert.alert('Success', 'Subscription added successfully')
      setName('')
      setAmount('')
      setCycle('Monthly')
      // Reload subscriptions
      await loadSubscriptions()
    } catch (error: any) {
      console.error('Error adding subscription:', error)
      Alert.alert('Error', `Failed to add subscription: ${error?.message || 'Unknown error'}`)
    } finally {
      setSaving(false)
    }
  }

  const handleRemoveSubscription = async (item: { id: string; db_id?: number; source?: string }) => {
    // Only allow deletion of user-added subscriptions
    if (item.source === 'user' && item.db_id) {
      try {
        await deleteSubscription(item.db_id, userId)
        Alert.alert('Success', 'Subscription removed successfully')
        // Reload subscriptions
        await loadSubscriptions()
      } catch (error: any) {
        console.error('Error deleting subscription:', error)
        Alert.alert('Error', `Failed to remove subscription: ${error?.message || 'Unknown error'}`)
      }
    } else {
      Alert.alert('Info', 'Auto-detected subscriptions cannot be deleted')
    }
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
          {['Weekly', 'Monthly', 'Yearly'].map(c => (
            <TouchableOpacity
              key={c}
              style={[
                styles.cycleButton,
                cycle === c && styles.cycleActive,
              ]}
              onPress={() =>
                setCycle(c as 'Weekly' | 'Monthly' | 'Yearly')
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
          style={[styles.addButton, saving && styles.addButtonDisabled]}
          onPress={handleAddSubscription}
          disabled={saving}
        >
          {saving ? (
            <ActivityIndicator color="#000" />
          ) : (
            <Text style={styles.addText}>Add Subscription</Text>
          )}
        </TouchableOpacity>
      </View>

      {/* Filter Tabs */}
      <View style={styles.filterRow}>
        {(['All', 'Weekly', 'Monthly', 'Yearly'] as const).map(c => (
          <TouchableOpacity
            key={c}
            style={[
              styles.filterChip,
              filterCycle === c && styles.filterChipActive,
            ]}
            onPress={() => setFilterCycle(c)}
          >
            <Text
              style={[
                styles.filterChipText,
                filterCycle === c && styles.filterChipTextActive,
              ]}
            >
              {c}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* Subscriptions List */}
      {loading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#F97316" />
          <Text style={styles.loadingText}>Loading subscriptions...</Text>
        </View>
      ) : (
        <FlatList
          data={
            filterCycle === 'All'
              ? subscriptions
              : subscriptions.filter(s => s.cycle === filterCycle)
          }
          keyExtractor={item => item.id}
          renderItem={({ item }) => (
            <View style={styles.subscriptionRow}>
              <View>
                <Text style={styles.subName}>{item.name}</Text>
                <Text style={styles.subCycle}>
                  {item.cycle} {item.source === 'detected' && '(Auto-detected)'}
                </Text>
              </View>

              <View style={styles.amountCol}>
                <Text style={styles.subAmount}>
                  ₹{item.amount}
                </Text>
                {item.source === 'user' && item.db_id && (
                  <TouchableOpacity
                    onPress={() => handleRemoveSubscription(item)}
                  >
                    <Text style={styles.remove}>Remove</Text>
                  </TouchableOpacity>
                )}
              </View>
            </View>
          )}
          ListEmptyComponent={
            <Text style={styles.emptyText}>No subscriptions yet</Text>
          }
        />
      )}

      {/* Done */}
      <TouchableOpacity
        style={styles.doneButton}
        onPress={() => navigation.navigate('index')}
      >
        <Text style={styles.doneText}>Done</Text>
      </TouchableOpacity>

      {/* Detected Subscription Modal */}
      <Modal
        visible={showDetectedModal}
        animationType="slide"
        transparent
        onRequestClose={() => setShowDetectedModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Recurring Transaction Detected</Text>
            <Text style={styles.modalText}>
              We detected a recurring transaction that might be a subscription:
            </Text>
            {detectedSubscription && (
              <View style={styles.detectedInfo}>
                <Text style={styles.detectedName}>{detectedSubscription.name}</Text>
                <Text style={styles.detectedAmount}>₹{detectedSubscription.amount}</Text>
                <Text style={styles.detectedCycle}>{detectedSubscription.cycle}</Text>
              </View>
            )}
            <Text style={styles.modalQuestion}>
              Would you like to add this as a subscription?
            </Text>
            <View style={styles.modalActions}>
              <TouchableOpacity
                style={styles.modalCancelBtn}
                onPress={() => {
                  setShowDetectedModal(false)
                  setDetectedSubscription(null)
                }}
              >
                <Text style={styles.modalCancelText}>Skip</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.modalAddBtn}
                onPress={async () => {
                  if (detectedSubscription) {
                    try {
                      setSaving(true)
                      let period: 'monthly' | 'yearly' | 'weekly' | 'daily' = 'monthly'
                      if (detectedSubscription.cycle === 'Yearly') period = 'yearly'
                      else if (detectedSubscription.cycle === 'Weekly') period = 'weekly'
                      else if (detectedSubscription.cycle === 'Daily') period = 'daily'
                      else period = 'monthly'
                      
                      await addSubscription(userId, detectedSubscription.name, detectedSubscription.amount, period)
                      Alert.alert('Success', 'Subscription added successfully')
                      setShowDetectedModal(false)
                      setDetectedSubscription(null)
                      await loadSubscriptions()
                    } catch (error: any) {
                      Alert.alert('Error', `Failed to add subscription: ${error?.message || 'Unknown error'}`)
                    } finally {
                      setSaving(false)
                    }
                  }
                }}
              >
                {saving ? (
                  <ActivityIndicator color="#000" />
                ) : (
                  <Text style={styles.modalAddText}>Add Subscription</Text>
                )}
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
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
  filterRow: {
    flexDirection: 'row',
    marginBottom: 12,
    marginTop: 4,
  },
  filterChip: {
    borderWidth: 1,
    borderColor: '#1F2933',
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 999,
    marginRight: 8,
    backgroundColor: '#0B0B0B',
  },
  filterChipActive: {
    backgroundColor: '#F97316',
    borderColor: '#F97316',
  },
  filterChipText: {
    color: '#9CA3AF',
    fontSize: 12,
    fontWeight: '500',
  },
  filterChipTextActive: {
    color: '#000',
    fontWeight: '600',
  },
  addButtonDisabled: {
    opacity: 0.6,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 40,
  },
  loadingText: {
    color: '#9CA3AF',
    marginTop: 12,
  },
  emptyText: {
    color: '#9CA3AF',
    textAlign: 'center',
    paddingVertical: 40,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.7)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    backgroundColor: '#111827',
    margin: 20,
    padding: 20,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#1F2933',
  },
  modalTitle: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 12,
  },
  modalText: {
    color: '#D1D5DB',
    fontSize: 14,
    marginBottom: 16,
  },
  detectedInfo: {
    backgroundColor: '#0B0B0B',
    padding: 16,
    borderRadius: 12,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#1F2933',
  },
  detectedName: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 4,
  },
  detectedAmount: {
    color: '#F97316',
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 4,
  },
  detectedCycle: {
    color: '#9CA3AF',
    fontSize: 12,
  },
  modalQuestion: {
    color: '#D1D5DB',
    fontSize: 14,
    marginBottom: 20,
  },
  modalActions: {
    flexDirection: 'row',
    gap: 12,
  },
  modalCancelBtn: {
    flex: 1,
    paddingVertical: 12,
    alignItems: 'center',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#1F2933',
  },
  modalCancelText: {
    color: '#9CA3AF',
    fontWeight: '600',
  },
  modalAddBtn: {
    flex: 1,
    backgroundColor: '#F97316',
    paddingVertical: 12,
    alignItems: 'center',
    borderRadius: 10,
  },
  modalAddText: {
    color: '#000',
    fontWeight: '600',
  },
})
