
import streamlit as st
from typing import Dict, List

def _render_rule_chips(rule_hits: List[str]):
    if not rule_hits:
        return
    cols = st.columns(len(rule_hits))
    for col, rule in zip(cols, rule_hits):
        
        col.markdown(
            f"<span style='background:#E8F0FF;border-radius:12px;padding:6px 10px;font-size:13px;'>"
            f"{rule}</span>",
            unsafe_allow_html=True,
        )

def _format_token_row(token: str, score: float):
    
    percent = int(round(score * 100))
    
    bar = (
        f"<div style='display:flex;align-items:center;'>"
        f"<div style='min-width:90px;padding-right:8px;font-size:14px'>{token}</div>"
        f"<div style='flex:1;margin-right:8px;height:12px;background:#EEE;border-radius:6px;overflow:hidden;'>"
        f"<div style='width:{percent}%;height:12px;background:linear-gradient(90deg,#4CAF50,#8BC34A);'></div>"
        f"</div>"
        f"<div style='width:48px;text-align:right;font-size:12px'>{percent}%</div>"
        f"</div>"
    )
    return bar

def render_explain_card(explanation: Dict):
   
    if not explanation:
        st.info("No explanation available.")
        return

    rule_hits = explanation.get("rule_hits") or []
    top_tokens = explanation.get("top_tokens") or []
    rationale = explanation.get("rationale") or ""

    st.write("**Explanation**")
    
    if rule_hits:
        _render_rule_chips(rule_hits)
    else:
        st.write("_No explicit rule hits_")

    
    try:
        sorted_tokens = sorted(top_tokens, key=lambda x: x.get("score", 0), reverse=True)
    except Exception:
        sorted_tokens = top_tokens

    st.write("")
    if sorted_tokens:
        for t in sorted_tokens:
            token = t.get("token", "<unknown>")
            score = float(t.get("score", 0.0) or 0.0)
            # token with tooltip showing exact score (2 decimals)
            bar_html = _format_token_row(token, score)
            st.markdown(f"<div title='score: {score:.4f}'>{bar_html}</div>", unsafe_allow_html=True)
    else:
        st.write("_No token-level explanation_")

    if rationale:
        st.write("**Rationale**")
        st.write(rationale)
