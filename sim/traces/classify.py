"""
Tool-call classification for real agent traces.

Splits tool calls into READ-ONLY vs EXTERNAL-EFFECT, and assigns each external
effect an irreversibility-relevant class. Used to test the paper's premise that
fleets of agents produce correlated bursts of external (often irreversible)
effects.

Classification is by an explicit allow/effect map derived from the full tool
vocabularies of tau-bench (retail+airline) and AgentDojo (banking/workspace/
travel/slack), with a conservative prefix fallback. See VOCAB_NOTES below.
"""

# --- External-effect tools -> (effect_class, irreversibility hint) ---------
# irreversibility hint: HIGH (money/irrevocable/disclosure), MED (state change,
# often compensatable), LOW (mostly refundable/revocable draft-like).
EXTERNAL = {
    # ---- tau-bench (retail + airline) ----
    "send_money":                 ("pay",     "HIGH"),   # (agentdojo banking; listed here for completeness)
    "return_delivered_order_items":("refund", "MED"),
    "modify_pending_order_items":  ("update",  "MED"),
    "exchange_delivered_order_items":("refund","MED"),
    "cancel_pending_order":        ("delete",  "MED"),
    "update_reservation_flights":  ("update",  "MED"),
    "transfer_to_human_agents":    ("send",    "LOW"),
    "modify_pending_order_address":("update",  "MED"),
    "cancel_reservation":          ("delete",  "MED"),
    "book_reservation":            ("create",  "HIGH"),  # charges a card
    "modify_user_address":         ("update",  "MED"),
    "update_reservation_baggages": ("update",  "MED"),
    "send_certificate":            ("pay",     "HIGH"),  # issues account credit
    "update_reservation_passengers":("update", "MED"),
    "modify_pending_order_payment":("pay",     "HIGH"),
    # ---- AgentDojo banking ----
    "schedule_transaction":        ("pay",     "HIGH"),
    "update_scheduled_transaction":("pay",     "HIGH"),
    "update_user_info":            ("update",  "MED"),
    "update_password":             ("update",  "HIGH"),
    # ---- AgentDojo workspace (email/calendar/cloud) ----
    "send_email":                  ("send",    "HIGH"),  # disclosure, irrevocable
    "delete_email":                ("delete",  "MED"),
    "create_calendar_event":       ("create",  "LOW"),
    "reschedule_calendar_event":   ("update",  "LOW"),
    "cancel_calendar_event":       ("delete",  "LOW"),
    "add_calendar_event_participants":("send",  "MED"),  # notifies + discloses
    "create_file":                 ("create",  "LOW"),
    "append_to_file":              ("update",  "LOW"),
    "delete_file":                 ("delete",  "MED"),
    "share_file":                  ("send",    "HIGH"),  # disclosure
    # ---- AgentDojo slack ----
    "send_direct_message":         ("send",    "HIGH"),  # disclosure
    "send_channel_message":        ("send",    "HIGH"),
    "post_webpage":                ("post",    "HIGH"),  # public disclosure
    "add_user_to_channel":         ("update",  "MED"),
    "remove_user_from_slack":      ("delete",  "MED"),
    "invite_user_to_slack":        ("send",    "MED"),
    # ---- AgentDojo travel ----
    "reserve_hotel":               ("create",  "HIGH"),
    "reserve_restaurant":          ("create",  "MED"),
    "reserve_car_rental":          ("create",  "HIGH"),
}

# Prefixes that are unambiguously read-only if not in EXTERNAL.
READONLY_PREFIXES = ("get_", "search_", "list_", "read_", "check_", "find_",
                     "calculate", "think")

def classify(name):
    """Return (is_external, effect_class or None, sev or None)."""
    if not name:
        return (False, None, None)
    if name in EXTERNAL:
        cls, sev = EXTERNAL[name]
        return (True, cls, sev)
    for p in READONLY_PREFIXES:
        if name.startswith(p):
            return (False, None, None)
    # Conservative fallback: verbs that mutate external state.
    for kw in ("send", "post", "delete", "create", "update", "modify",
               "cancel", "book", "reserve", "pay", "transfer", "share",
               "schedule", "remove", "add_", "invite", "append", "exchange",
               "return_", "deploy"):
        if kw in name:
            return (True, "other", "MED")
    return (False, None, None)  # unknown read-like default
