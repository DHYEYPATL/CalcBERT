"""
LangGraph Workflow Orchestrator for v2 Backend.
Orchestrates multi-step prepay check workflow with sequential and optional LangGraph modes.
"""

from typing import Dict, Any, Optional
from backend.config import settings
from backend.model_adapter import ModelAdapter
from backend.policy_engine import check_policy, get_policy_options
import backend.vector_store as vector_store

# Try to import LangGraph (optional)
try:
    from langgraph.graph import Graph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


class WorkflowOrchestrator:
    """
    Orchestrates the prepay check workflow.
    Supports both sequential and LangGraph modes.
    """
    
    def __init__(self):
        self.model_adapter = ModelAdapter()
        self.mode = "langgraph" if (settings.LANGGRAPH_ENABLED and LANGGRAPH_AVAILABLE) else "sequential"
        print(f"✓ Workflow orchestrator initialized (mode: {self.mode})")
    
    def run_workflow(self, state: Dict[str, Any], mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the prepay check workflow.
        
        Args:
            state: Initial state with merchant, amount, note, user_role, etc.
            mode: Override workflow mode ('sequential' or 'langgraph')
            
        Returns:
            Final state with all analysis results
        """
        # Use specified mode or default
        workflow_mode = mode or self.mode
        
        if workflow_mode == "langgraph" and LANGGRAPH_AVAILABLE:
            return self._run_langgraph_workflow(state)
        else:
            return self._run_sequential_workflow(state)
    
    def _run_sequential_workflow(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run workflow in sequential mode (deterministic execution).
        
        Args:
            state: Initial state
            
        Returns:
            Final state with all results
        """
        # Node 1: Merchant Check
        state = self._merchant_check_node(state)
        
        # Node 2: TF-IDF Prediction
        state = self._tfidf_node(state)
        
        # Node 3: DistilBERT Prediction (conditional)
        if self.model_adapter.distil is not None:
            state = self._bert_node(state)
        
        # Node 4: Policy Engine
        state = self._policy_node(state)
        
        # Node 5: Fusion
        state = self._fusion_node(state)
        
        # Node 6: Explanation Builder
        state = self._explain_node(state)
        
        return state
    
    def _run_langgraph_workflow(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run workflow using LangGraph (advanced graph-based orchestration).
        
        Args:
            state: Initial state
            
        Returns:
            Final state with all results
        """
        # For now, fall back to sequential
        # In production, this would build a LangGraph graph
        print("ℹ LangGraph mode requested but using sequential for now")
        return self._run_sequential_workflow(state)
    
    def _merchant_check_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Node 1: Merchant verification and metadata enrichment.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with merchant_status
        """
        merchant = state.get("merchant", "")
        amount = state.get("amount", 0.0)
        note = state.get("note", "")
        
        # Detect merchant type
        merchant_type = self.model_adapter.detect_merchant_type(merchant)
        
        # Detect subscription
        subscription_info = self.model_adapter.detect_subscription(merchant, amount, note)
        
        # Check note quality
        note_quality = self.model_adapter.detect_low_quality_note(note)
        
        state["merchant_status"] = {
            "merchant_type": merchant_type,
            "is_verified": merchant_type == "verified",
            "subscription": subscription_info,
            "note_quality": note_quality
        }
        
        return state
    
    def _tfidf_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Node 2: TF-IDF category prediction.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with tfidf_output
        """
        predictions = self.model_adapter.predict_structured(state)
        state["tfidf_output"] = predictions.get("tfidf_output")
        state["rule_output"] = predictions.get("rule_output")
        
        return state
    
    def _bert_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Node 3: DistilBERT category prediction (optional).
        
        Args:
            state: Current state
            
        Returns:
            Updated state with bert_output
        """
        predictions = self.model_adapter.predict_structured(state)
        state["bert_output"] = predictions.get("bert_output")
        
        return state
    
    def _policy_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Node 4: Policy engine check.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with policy_result
        """
        user_role = state.get("user_role", "employee")
        amount = state.get("amount", 0.0)
        
        # Get category from predictions (prefer rule, then tfidf, then bert)
        category = "Unknown"
        if state.get("rule_output") and state["rule_output"].get("label"):
            category = state["rule_output"]["label"]
        elif state.get("tfidf_output") and state["tfidf_output"].get("label"):
            category = state["tfidf_output"]["label"]
        elif state.get("bert_output") and state["bert_output"].get("label"):
            category = state["bert_output"]["label"]
        
        # Get merchant type
        merchant_status = state.get("merchant_status", {})
        merchant_type = merchant_status.get("merchant_type", "unknown")
        
        # Build flags
        flags = {
            "is_subscription": merchant_status.get("subscription", {}).get("is_subscription", False),
            "subscription_approved": False,  # Would check database in production
            "low_quality_note": merchant_status.get("note_quality", {}).get("is_low_quality", False)
        }
        
        # Check policy
        policy_result = check_policy(user_role, category, amount, merchant_type, flags)
        
        # Add options
        policy_result["options"] = get_policy_options(policy_result["action"])
        
        state["policy_result"] = policy_result
        
        return state
    
    def _fusion_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Node 5: Fuse predictions and policy results.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with fusion_result
        """
        rule_output = state.get("rule_output")
        tfidf_output = state.get("tfidf_output")
        bert_output = state.get("bert_output")
        
        # Use fusion module if available
        if self.model_adapter.fusion:
            try:
                fused = self.model_adapter.fusion.fuse(rule_output, None, tfidf_output)
            except Exception as e:
                print(f"Fusion error: {e}, using fallback")
                fused = self._fallback_fusion(rule_output, tfidf_output, bert_output)
        else:
            fused = self._fallback_fusion(rule_output, tfidf_output, bert_output)
        
        state["fusion_result"] = fused
        
        # Determine final decision based on policy
        policy_result = state.get("policy_result", {})
        state["final_decision"] = policy_result.get("action", "allow")
        
        return state
    
    def _fallback_fusion(self, rule_output, tfidf_output, bert_output) -> Dict[str, Any]:
        """
        Fallback fusion logic when fusion module unavailable.
        
        Returns:
            Fused prediction
        """
        # Prefer rule if high confidence
        if rule_output and rule_output.get("confidence", 0) >= 0.9:
            return {
                "label": rule_output["label"],
                "confidence": rule_output["confidence"],
                "model_used": "rule",
                "rationale": {"rule_hits": rule_output.get("matches", [])}
            }
        
        # Otherwise prefer TF-IDF
        if tfidf_output:
            return {
                "label": tfidf_output.get("label", "Unknown"),
                "confidence": tfidf_output.get("confidence", 0.0),
                "model_used": "tfidf",
                "rationale": {"top_tokens": tfidf_output.get("top_tokens", [])}
            }
        
        # Fallback to bert
        if bert_output:
            return {
                "label": bert_output.get("label", "Unknown"),
                "confidence": bert_output.get("confidence", 0.0),
                "model_used": "bert",
                "rationale": {}
            }
        
        # No predictions available
        return {
            "label": "Unknown",
            "confidence": 0.0,
            "model_used": "none",
            "rationale": {}
        }
    
    def _explain_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Node 6: Build human-readable explanation.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with explanation
        """
        fusion_result = state.get("fusion_result", {})
        policy_result = state.get("policy_result", {})
        merchant_status = state.get("merchant_status", {})
        
        # Build explanation
        explanation = {
            "category_reason": self._build_category_explanation(fusion_result),
            "policy_reason": self._build_policy_explanation(policy_result),
            "suggestions": self._build_suggestions(state)
        }
        
        state["explanation"] = explanation
        
        return state
    
    def _build_category_explanation(self, fusion_result: Dict[str, Any]) -> str:
        """Build category prediction explanation."""
        label = fusion_result.get("label", "Unknown")
        confidence = fusion_result.get("confidence", 0.0)
        model_used = fusion_result.get("model_used", "none")
        
        if model_used == "rule":
            return f"Categorized as '{label}' based on keyword matching (high confidence: {confidence:.0%})"
        elif model_used == "tfidf":
            return f"Categorized as '{label}' using ML model (confidence: {confidence:.0%})"
        elif model_used == "bert":
            return f"Categorized as '{label}' using advanced AI model (confidence: {confidence:.0%})"
        else:
            return "Unable to determine category"
    
    def _build_policy_explanation(self, policy_result: Dict[str, Any]) -> str:
        """Build policy decision explanation."""
        from backend.policy_engine import format_policy_explanation
        
        action = policy_result.get("action", "allow")
        reasons = policy_result.get("policy_reasons", [])
        
        if action == "allow" and not reasons:
            return "No policy restrictions apply."
        
        return format_policy_explanation(action, reasons)
    
    def _build_suggestions(self, state: Dict[str, Any]) -> list:
        """Build actionable suggestions."""
        suggestions = []
        
        merchant_status = state.get("merchant_status", {})
        note_quality = merchant_status.get("note_quality", {})
        
        # Add note quality suggestions
        if note_quality.get("is_low_quality"):
            suggestions.extend(note_quality.get("suggestions", []))
        
        # Add policy-based suggestions
        policy_result = state.get("policy_result", {})
        action = policy_result.get("action")
        
        if action == "warn":
            suggestions.append("Consider adding more context to justify this expense")
        elif action == "block":
            suggestions.append("This expense violates company policy - please contact your manager")
        
        return suggestions


# Global orchestrator instance
_orchestrator = None


def get_orchestrator() -> WorkflowOrchestrator:
    """
    Get the global workflow orchestrator instance.
    
    Returns:
        WorkflowOrchestrator instance
    """
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = WorkflowOrchestrator()
    return _orchestrator


def run_workflow(state: Dict[str, Any], mode: Optional[str] = None) -> Dict[str, Any]:
    """
    Run the prepay check workflow (convenience function).
    
    Args:
        state: Initial state
        mode: Optional workflow mode override
        
    Returns:
        Final state with all results
    """
    orchestrator = get_orchestrator()
    return orchestrator.run_workflow(state, mode)
