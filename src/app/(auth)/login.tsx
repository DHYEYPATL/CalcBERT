import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  KeyboardAvoidingView,
  Platform,
} from "react-native";
import React, { useState } from "react";
import { SafeAreaView } from "react-native-safe-area-context";
import { useNavigation } from "@react-navigation/native";
import { Ionicons } from "@expo/vector-icons";
import { router } from "expo-router";

const LoginScreen = () => {
  const navigation = useNavigation<any>();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

const handleLogin = () => {
  router.replace("/(main)");
};

  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView
        behavior={Platform.OS === "ios" ? "padding" : undefined}
        style={{ flex: 1 }}
      >
        {/* HEADER / BRAND */}
        <View style={styles.header}>
          <Ionicons name="wallet-outline" size={42} color="#F97316" />
          <Text style={styles.appName}>CalcBERT</Text>
          <Text style={styles.tagline}>
            Smarter payments. Safer decisions.
          </Text>
        </View>

        {/* LOGIN CARD */}
        <View style={styles.card}>
          <Text style={styles.title}>Login</Text>

          {/* EMAIL */}
          <View style={styles.inputWrapper}>
            <Ionicons name="mail-outline" size={18} color="#9CA3AF" />
            <TextInput
              style={styles.input}
              placeholder="Email"
              placeholderTextColor="#6B7280"
              value={email}
              onChangeText={setEmail}
              autoCapitalize="none"
            />
          </View>

          {/* PASSWORD */}
          <View style={styles.inputWrapper}>
            <Ionicons name="lock-closed-outline" size={18} color="#9CA3AF" />
            <TextInput
              style={styles.input}
              placeholder="Password"
              placeholderTextColor="#6B7280"
              secureTextEntry
              value={password}
              onChangeText={setPassword}
            />
          </View>

          {/* LOGIN BUTTON */}
          <TouchableOpacity
            style={[
              styles.loginButton,
              (!email || !password) && styles.disabled,
            ]}
            disabled={!email || !password}
            onPress={handleLogin}
          >
            <Text style={styles.loginText}>Login</Text>
          </TouchableOpacity>

          {/* EXTRA ACTIONS */}
          <View style={styles.footerRow}>
            <TouchableOpacity>
              <Text style={styles.linkText}>Forgot Password?</Text>
            </TouchableOpacity>

            <TouchableOpacity>
              <Text style={styles.linkText}>Sign Up</Text>
            </TouchableOpacity>
          </View>
        </View>

        {/* FOOTER */}
        <View style={styles.bottomTextContainer}>
          <Text style={styles.bottomText}>
            By continuing, you agree to our{" "}
            <Text style={styles.linkText}>Terms & Privacy Policy</Text>
          </Text>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
};

export default LoginScreen;

/* ================= STYLES ================= */

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#0B0B0B",
  },

  header: {
    alignItems: "center",
    marginTop: 40,
    marginBottom: 32,
  },

  appName: {
    color: "#FFFFFF",
    fontSize: 26,
    fontWeight: "700",
    marginTop: 8,
  },

  tagline: {
    color: "#9CA3AF",
    fontSize: 13,
    marginTop: 4,
  },

  card: {
    backgroundColor: "#111827",
    marginHorizontal: 24,
    padding: 20,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "#1F2933",
  },

  title: {
    color: "#FFFFFF",
    fontSize: 20,
    fontWeight: "600",
    marginBottom: 20,
    textAlign: "center",
  },

  inputWrapper: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#0B0B0B",
    borderRadius: 12,
    paddingHorizontal: 14,
    paddingVertical: 12,
    marginBottom: 14,
    borderWidth: 1,
    borderColor: "#1F2933",
  },

  input: {
    flex: 1,
    color: "#FFFFFF",
    fontSize: 14,
    marginLeft: 10,
  },

  loginButton: {
    backgroundColor: "#F97316",
    borderRadius: 14,
    paddingVertical: 16,
    alignItems: "center",
    marginTop: 8,
  },

  disabled: {
    backgroundColor: "#6B7280",
  },

  loginText: {
    color: "#000",
    fontSize: 16,
    fontWeight: "600",
  },

  footerRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 16,
  },

  linkText: {
    color: "#F97316",
    fontSize: 13,
    fontWeight: "500",
  },

  bottomTextContainer: {
    marginTop: "auto",
    padding: 16,
  },

  bottomText: {
    color: "#9CA3AF",
    fontSize: 12,
    textAlign: "center",
  },
});
