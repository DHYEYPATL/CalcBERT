import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  TextInput,
} from 'react-native'
import React, { useState } from 'react'
import { SafeAreaView } from 'react-native-safe-area-context'
import { useRoute, useNavigation } from '@react-navigation/native'

const PaymentScreen = () => {
  const route = useRoute<any>()
  const navigation = useNavigation<any>()

  const merchantName = route.params?.merchantName
  const upiId = route.params?.upiId

  const [amount, setAmount] = useState('')
  const [note, setNote] = useState('')

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity style={styles.iconWrapper}>
          <Text style={styles.iconText}>←</Text>
        </TouchableOpacity>

        <Text style={styles.paytext}>Pay via UPI</Text>

        <TouchableOpacity
          style={styles.iconWrapper}
          onPress={() => navigation.navigate('QRScannerScreen')}
        >
          <Text style={styles.iconText}>⌁</Text>
        </TouchableOpacity>
      </View>

      {/* Merchant + Amount + Note (Only after QR scan) */}
      {merchantName && (
        <>
          {/* Merchant Card */}
          <View style={styles.merchantCard}>
            <Text style={styles.merchantName}>{merchantName}</Text>
            <Text style={styles.upiText}>{upiId}</Text>

            <View style={styles.verifiedRow}>
              <Text style={styles.verifiedIcon}>✔</Text>
              <Text style={styles.verifiedText}>Verified Merchant</Text>
            </View>
          </View>

          {/* Amount Section */}
          <View style={styles.amountSection}>
            <Text style={styles.currency}>₹</Text>
            <TextInput
              style={styles.amountInput}
              placeholder="0"
              placeholderTextColor="#6B7280"
              keyboardType="numeric"
              value={amount}
              onChangeText={setAmount}
            />
          </View>

          {/* Note Input */}
          <View style={styles.noteSection}>
            <TextInput
              style={styles.noteInput}
              placeholder="Add a note (optional)"
              placeholderTextColor="#6B7280"
              value={note}
              onChangeText={setNote}
            />
          </View>
        </>
      )}

      {/* Proceed Button */}
      {merchantName && (
        <View style={styles.footer}>
          <TouchableOpacity
            style={[
              styles.proceedButton,
              !amount && styles.proceedDisabled,
            ]}
            disabled={!amount}
            onPress={() => navigation.navigate('PrepayResultScreen', {
              merchantName,
              upiId,
              amount,
              note,
            })}
          >
            <Text style={styles.proceedText}>Proceed</Text>
          </TouchableOpacity>
        </View>
      )}
    </SafeAreaView>
  )
}

export default PaymentScreen;
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0B0B0B',
  },

  header: {
    height: 64,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#1F2933',
  },

  paytext: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '600',
    letterSpacing: 1,
  },

  iconWrapper: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },

  iconText: {
    color: '#F97316',
    fontSize: 26,
    fontWeight: '600',
  },

  merchantCard: {
    backgroundColor: '#111827',
    margin: 16,
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#1F2933',
  },

  merchantName: {
    color: '#FFFFFF',
    fontSize: 20,
    fontWeight: '600',
  },

  upiText: {
    color: '#9CA3AF',
    marginTop: 4,
    fontSize: 14,
  },

  verifiedRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 12,
  },

  verifiedIcon: {
    color: '#22C55E',
    marginRight: 6,
  },

  verifiedText: {
    color: '#22C55E',
    fontSize: 14,
  },

  amountSection: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 24,
  },

  currency: {
    color: '#FFFFFF',
    fontSize: 36,
    marginRight: 6,
  },

  amountInput: {
    color: '#FFFFFF',
    fontSize: 36,
    fontWeight: '600',
    minWidth: 120,
    textAlign: 'center',
    borderBottomWidth: 2,
    borderBottomColor: '#F97316',
  },

  noteSection: {
    marginHorizontal: 32,
    marginTop: 24,
  },

  noteInput: {
    backgroundColor: '#111827',
    borderRadius: 10,
    paddingHorizontal: 14,
    paddingVertical: 12,
    color: '#FFFFFF',
    fontSize: 14,
    borderWidth: 1,
    borderColor: '#1F2933',
  },

  footer: {
    marginTop: 'auto',
    padding: 16,
  },

  proceedButton: {
    backgroundColor: '#F97316',
    borderRadius: 14,
    paddingVertical: 16,
    alignItems: 'center',
  },

  proceedDisabled: {
    backgroundColor: '#6B7280',
  },

  proceedText: {
    color: '#000',
    fontSize: 16,
    fontWeight: '600',
  },
})
