import { View, Text, StyleSheet, TouchableOpacity } from 'react-native'
import React from 'react'
import { SafeAreaView } from 'react-native-safe-area-context'
import { useNavigation } from '@react-navigation/native'
import { useRoute } from '@react-navigation/native'


const DashboardScreen = () => {
  const navigation = useNavigation<any>()
  const route = useRoute<any>()

const merchantName = route.params?.merchantName
const upiId = route.params?.upiId


  return (
    <SafeAreaView style={styles.container}>
      
      {/* Header */}
      <View style={styles.header}>
        
        {/* Back Arrow */}
        <TouchableOpacity style={styles.iconWrapper}>
          <Text style={styles.iconText}>←</Text>
        </TouchableOpacity>

        {/* Title */}
        <Text style={styles.paytext}>Pay via UPI</Text>

        {/* QR Scan */}
        <TouchableOpacity
          style={styles.iconWrapper}
          onPress={() => navigation.navigate('QRScannerScreen')}
        >
          <Text style={styles.iconText}>⌁</Text>
        </TouchableOpacity>

      </View>
      {merchantName && (
  <View style={styles.merchantCard}>
    <Text style={styles.merchantName}>{merchantName}</Text>
    <Text style={styles.upiText}>{upiId}</Text>

    <View style={styles.verifiedRow}>
      <Text style={styles.verifiedIcon}>✔</Text>
      <Text style={styles.verifiedText}>Verified Merchant</Text>
    </View>
  </View>
)}

    </SafeAreaView>
  )
}

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
    fontSize: 28,
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
})

export default DashboardScreen
