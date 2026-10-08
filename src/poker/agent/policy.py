"""Compatibility alias for the original action-producing interface.

New LLM policies use poker.agent.react.policy.Policy. The complete decision
module implements DecisionEngine; this alias keeps existing baseline factories
and older integrations working.
"""

from poker.agent.decision_engine import DecisionEngine as Policy

__all__ = ["Policy"]
