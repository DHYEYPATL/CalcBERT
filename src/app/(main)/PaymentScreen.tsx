import { View, Text } from 'react-native'
import React from 'react'
import { useState } from 'react';

const PaymentScreen = () => {
const [merchant, setMerchant] = useState(null);
const [amount, setAmount] = useState('');
const [note, setNote] = useState('');
const [upiId, setUpiId] = useState('');
const [role, setRole] = useState(null);

const [loading, setLoading] = useState(false);
const [error, setError] = useState(null);

  return (
    <View>
      <Text>PaymentScreen</Text>
    </View>
  )
}

export default PaymentScreen