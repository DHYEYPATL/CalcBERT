import { View, Text, StyleSheet, TouchableOpacity } from 'react-native'
import React, { useEffect, useState } from 'react'
import { CameraView, useCameraPermissions } from 'expo-camera'
import { SafeAreaView } from 'react-native-safe-area-context'
import { useNavigation } from '@react-navigation/native'

const QRScannerScreen = () => {
  const [permission, requestPermission] = useCameraPermissions()
  const [scanned, setScanned] = useState(false)
  const navigation = useNavigation<any>()

  useEffect(() => {
    requestPermission()
  }, [])

  if (!permission || !permission.granted) {
    return (
      <SafeAreaView style={styles.permission}>
        <Text style={styles.permissionText}>
          Camera permission required
        </Text>
      </SafeAreaView>
    )
  }

const handleScan = ({ data }: { data: string }) => {
  if (scanned) return
  setScanned(true)

  try {
    const url = new URL(data)
    const upiId = url.searchParams.get('pa')
    const merchantName = url.searchParams.get('pn')

    navigation.navigate('index', {
      merchantName: merchantName
        ? decodeURIComponent(merchantName)
        : 'Unknown Merchant',
      upiId: upiId ?? '',
    })
  } catch (err) {
    console.log('Invalid QR')
    navigation.goBack()
  }
}

  return (
    <SafeAreaView style={styles.container}>
      <CameraView
        style={StyleSheet.absoluteFill}
        onBarcodeScanned={handleScan}
        barcodeScannerSettings={{
          barcodeTypes: ['qr'],
        }}
      />

      {/* Overlay */}
      <View style={styles.overlay}>
        <View style={styles.scanBox} />
        <Text style={styles.scanText}>
          Align QR code within the frame
        </Text>

        <TouchableOpacity
          style={styles.cancel}
          onPress={() => navigation.goBack()}
        >
          <Text style={styles.cancelText}>Cancel</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },

  overlay: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },

  scanBox: {
    width: 220,
    height: 220,
    borderWidth: 2,
    borderColor: '#F97316',
    borderRadius: 12,
  },

  scanText: {
    color: '#FFFFFF',
    marginTop: 20,
    fontSize: 14,
  },

  cancel: {
    position: 'absolute',
    bottom: 40,
  },

  cancelText: {
    color: '#F97316',
    fontSize: 16,
  },

  permission: {
    flex: 1,
    backgroundColor: '#000',
    justifyContent: 'center',
    alignItems: 'center',
  },

  permissionText: {
    color: '#FFF',
  },
})

export default QRScannerScreen
