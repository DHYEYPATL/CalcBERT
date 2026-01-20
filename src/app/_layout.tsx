import { View, Text } from 'react-native'
import React, { useState } from 'react'
import { Redirect, Stack } from 'expo-router';

const _layout = () => {
    const [isLogin, setisLogin] = useState(false);
  return (
    <Stack screenOptions={{ headerShown: false }}>
      {isLogin && <Stack.Screen name="(main)" />}
      {!isLogin && <Stack.Screen name="(auth)" />}
    </Stack>
  )
}

export default _layout
