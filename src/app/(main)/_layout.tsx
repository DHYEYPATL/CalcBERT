import { View, Text } from 'react-native'
import React from 'react'
import { Stack } from 'expo-router'
import QRScannerScreen from './QRScannerScreen'

const _layout = () => {
  return (
    <Stack screenOptions={{headerShown:false}}>
      <Stack.Screen name='index'/>
      <Stack.Screen name='QRScannerScreen'/>
      <Stack.Screen name='PaymentScreen'/>
      <Stack.Screen name='PrepayResultScreen'/>
      <Stack.Screen name='SplitPaymentScreen' />
    </Stack>
  )
}

export default _layout