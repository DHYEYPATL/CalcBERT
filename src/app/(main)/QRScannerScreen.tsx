import { useNavigation } from '@react-navigation/native'
import { CameraView, useCameraPermissions } from 'expo-camera'
import { useEffect, useState } from 'react'
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'
import { Ionicons } from '@expo/vector-icons'

const QRScannerScreen = () => {
  const [permission, requestPermission] = useCameraPermissions()
  const [scanned, setScanned] = useState(false)
  const navigation = useNavigation<any>()

  useEffect(() => {
    requestPermission()
  }, [])

  const handleScan = ({ data }: { data: string }) => {
    if (scanned) return
    setScanned(true)
    try {
      const url = new URL(data)
      const upiId = url.searchParams.get('pa')
      const merchantName = url.searchParams.get('pn')
      navigation.replace('PaymentScreen', {
        merchantName: merchantName ? decodeURIComponent(merchantName) : 'Unknown Merchant',
        upiId: upiId ?? '',
      })
    } catch {
      navigation.goBack()
    }
  }

  if (!permission || !permission.granted) {
    return (
      <SafeAreaView style={styles.permission}>
        <Ionicons name="camera-outline" size={56} color="#9CA3AF" />
        <Text style={styles.permissionText}>Camera permission required to scan UPI QR</Text>
        <TouchableOpacity style={styles.reqBtn} onPress={() => requestPermission()}>
          <Text style={styles.reqBtnText}>Grant access</Text>
        </TouchableOpacity>
      </SafeAreaView>
    )
  }

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()} style={styles.headerBtn}>
          <Ionicons name="arrow-back" size={24} color="#FFF" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Scan UPI QR</Text>
        <View style={styles.headerBtn} />
      </View>

      <CameraView
        style={StyleSheet.absoluteFillObject}
        onBarcodeScanned={scanned ? undefined : handleScan}
        barcodeScannerSettings={{ barcodeTypes: ['qr'] }}
      />

      {/* Dark overlay with cutout */}
      <View style={styles.overlay} pointerEvents="box-none">
        <View style={styles.scanBox} />
        <Text style={styles.scanText}>Position the UPI QR code inside the frame</Text>

        <View style={styles.footer}>
          <TouchableOpacity style={styles.cancelBtn} onPress={() => navigation.goBack()}>
            <Ionicons name="close" size={22} color="#FFF" />
            <Text style={styles.cancelText}>Cancel</Text>
          </TouchableOpacity>
        </View>
      </View>
    </SafeAreaView>
  )
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0B0B0B' },

  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 12,
    paddingVertical: 10,
    backgroundColor: 'rgba(0,0,0,0.6)',
  },
  headerBtn: { width: 40, height: 40, justifyContent: 'center', alignItems: 'center' },
  headerTitle: { color: '#FFF', fontSize: 18, fontWeight: '600' },

  overlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
  },

  scanBox: {
    width: 240,
    height: 240,
    borderWidth: 3,
    borderColor: '#F97316',
    borderRadius: 16,
    backgroundColor: 'transparent',
  },

  scanText: {
    color: 'rgba(255,255,255,0.9)',
    marginTop: 24,
    fontSize: 15,
    paddingHorizontal: 24,
    textAlign: 'center',
  },

  footer: { position: 'absolute', bottom: 40, left: 0, right: 0, alignItems: 'center' },
  cancelBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255,255,255,0.15)',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 12,
    gap: 8,
  },
  cancelText: { color: '#FFF', fontSize: 16, fontWeight: '500' },

  permission: {
    flex: 1,
    backgroundColor: '#0B0B0B',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  permissionText: { color: '#9CA3AF', fontSize: 16, textAlign: 'center', marginTop: 16 },
  reqBtn: {
    marginTop: 24,
    paddingVertical: 12,
    paddingHorizontal: 24,
    backgroundColor: '#F97316',
    borderRadius: 12,
  },
  reqBtnText: { color: '#000', fontWeight: '600', fontSize: 16 },
})

export default QRScannerScreen
