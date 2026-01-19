"""
Quick test script for LangGraph workflow implementation.
"""

import sys
sys.path.insert(0, '.')

from backend.langgraph_workflow import WorkflowOrchestrator

def test_langgraph():
    print("="*60)
    print("Testing LangGraph Workflow Implementation")
    print("="*60)
    
    # Create orchestrator
    orchestrator = WorkflowOrchestrator()
    print(f"\n✓ Orchestrator initialized in '{orchestrator.mode}' mode")
    
    # Test state
    test_state = {
        "merchant": "Swiggy",
        "amount": 1200.0,
        "note": "client meeting dinner",
        "upi_id": "swiggy@upi",
        "user_role": "employee",
        "user_id": "test-user",
        "date": "2026-01-19"
    }
    
    print(f"\n📝 Test input:")
    print(f"   Merchant: {test_state['merchant']}")
    print(f"   Amount: ₹{test_state['amount']}")
    print(f"   Note: {test_state['note']}")
    print(f"   Role: {test_state['user_role']}")
    
    # Run workflow
    print(f"\n🚀 Running LangGraph workflow...")
    result = orchestrator.run_workflow(test_state, mode="langgraph")
    
    # Display results
    print(f"\n✅ Workflow completed!")
    print(f"\n📊 Results:")
    print(f"   Final Decision: {result.get('final_decision', 'N/A')}")
    
    fusion = result.get('fusion_result', {})
    print(f"   Category: {fusion.get('label', 'N/A')}")
    print(f"   Confidence: {fusion.get('confidence', 0):.2%}")
    print(f"   Model Used: {fusion.get('model_used', 'N/A')}")
    
    policy = result.get('policy_result', {})
    print(f"   Policy Action: {policy.get('action', 'N/A')}")
    
    merchant_status = result.get('merchant_status', {})
    print(f"   Merchant Type: {merchant_status.get('merchant_type', 'N/A')}")
    
    print("\n" + "="*60)
    print("✓ LangGraph test completed successfully!")
    print("="*60)

if __name__ == "__main__":
    test_langgraph()
