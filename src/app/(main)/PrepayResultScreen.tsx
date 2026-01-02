import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
} from 'react-native'
import React from 'react'
import { SafeAreaView } from 'react-native-safe-area-context'
import { useRoute, useNavigation } from '@react-navigation/native'

const PrePaymentScreen = () => {
  const route = useRoute<any>()
  const navigation = useNavigation<any>()

  const { merchantName, amount, note } = route.params || {}

  // ---- MOCK CalcBERT RESULTS (replace with API later) ----
  const category = 'Electronics / Retail'
  const confidence = 82 // %
  const trustScore = 7.8 // /10
  const riskWarnings = [
    'Merchant account created recently',
    'Transaction amount slightly higher than usual',
  ]

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Text style={styles.backText}>←</Text>
        </TouchableOpacity>
        <Text style={styles.title}>Pre-Payment Check</Text>
        <View style={{ width: 24 }} />
      </View>

      {/* Merchant Summary */}
      <View style={styles.card}>
        <Text style={styles.merchantName}>{merchantName}</Text>
        <Text style={styles.amountText}>₹ {amount}</Text>
        {note ? <Text style={styles.noteText}>“{note}”</Text> : null}
      </View>

      {/* Category Prediction */}
      <View style={styles.card}>
        <Text style={styles.sectionLabel}>Category Prediction</Text>
        <Text style={styles.sectionValue}>{category}</Text>
      </View>

      {/* Confidence Meter */}
      <View style={styles.card}>
        <Text style={styles.sectionLabel}>Confidence Meter</Text>

        <View style={styles.meterBackground}>
          <View
            style={[
              styles.meterFill,
              { width: `${confidence}%` },
            ]}
          />
        </View>

        <Text style={styles.confidenceText}>
          {confidence}% confidence this is a legitimate payment
        </Text>
      </View>

      {/* Risk Warnings */}
      <View style={styles.card}>
        <Text style={styles.sectionLabel}>Risk Warnings</Text>

        {riskWarnings.map((risk, index) => (
          <View key={index} style={styles.warningRow}>
            <Text style={styles.warningIcon}>⚠</Text>
            <Text style={styles.warningText}>{risk}</Text>
          </View>
        ))}
      </View>

      {/* Trust Status */}
      <View style={styles.card}>
        <Text style={styles.sectionLabel}>Merchant Trust Status</Text>
        <Text style={styles.trustText}>
          Trust Score: {trustScore} / 10
        </Text>
      </View>

      {/* Action Buttons */}
      <View style={styles.footer}>
        <TouchableOpacity style={styles.primaryButton}>
          <Text style={styles.primaryText}>Continue Anyway</Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.secondaryButton}>
          <Text style={styles.secondaryText}>Edit Note</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.cancelButton}
          onPress={() => navigation.goBack()}
        >
          <Text style={styles.cancelText}>Cancel</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  )
}

export default PrePaymentScreen
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

  backText: {
    color: '#F97316',
    fontSize: 22,
  },

  title: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '600',
  },

  card: {
    backgroundColor: '#111827',
    marginHorizontal: 16,
    marginTop: 16,
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#1F2933',
  },

  merchantName: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '600',
  },

  amountText: {
    color: '#F97316',
    fontSize: 22,
    marginTop: 6,
    fontWeight: '600',
  },

  noteText: {
    color: '#9CA3AF',
    marginTop: 6,
    fontSize: 14,
  },

  sectionLabel: {
    color: '#9CA3AF',
    fontSize: 13,
  },

  sectionValue: {
    color: '#FFFFFF',
    fontSize: 16,
    marginTop: 4,
  },

  meterBackground: {
    height: 8,
    backgroundColor: '#1F2933',
    borderRadius: 6,
    marginTop: 10,
  },

  meterFill: {
    height: 8,
    backgroundColor: '#F97316',
    borderRadius: 6,
  },

  confidenceText: {
    color: '#9CA3AF',
    fontSize: 13,
    marginTop: 8,
  },

  warningRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 8,
  },

  warningIcon: {
    color: '#FB923C',
    marginRight: 8,
  },

  warningText: {
    color: '#FFFFFF',
    fontSize: 14,
    flex: 1,
  },

  trustText: {
    color: '#22C55E',
    fontSize: 16,
    marginTop: 6,
    fontWeight: '600',
  },

  footer: {
    marginTop: 'auto',
    padding: 16,
  },

  primaryButton: {
    backgroundColor: '#F97316',
    paddingVertical: 16,
    borderRadius: 14,
    alignItems: 'center',
    marginBottom: 12,
  },

  primaryText: {
    color: '#000',
    fontSize: 16,
    fontWeight: '600',
  },

  secondaryButton: {
    borderWidth: 1,
    borderColor: '#F97316',
    paddingVertical: 14,
    borderRadius: 14,
    alignItems: 'center',
    marginBottom: 12,
  },

  secondaryText: {
    color: '#F97316',
    fontSize: 15,
    fontWeight: '500',
  },

  cancelButton: {
    alignItems: 'center',
    paddingVertical: 8,
  },

  cancelText: {
    color: '#9CA3AF',
    fontSize: 14,
  },
})
