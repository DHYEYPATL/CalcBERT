import { View, Text } from 'react-native'
import React from 'react'
import { Stack } from 'expo-router';

const _layout = () => {
    const [isLogin, setisLogin] = useState(false);
  return (
    <Stack screenOptions={{ headerShown: false }}>
      <Stack.Screen name="(auth)" />
      <Stack.Screen name="(main)" />
    </Stack>
  )
}

export default _layout
