import pandas as pd

from stage4.simulator_v3.enums import EventType
from stage4.simulator_v3.event_queue import EVENT_PRIORITY, EventQueue


def test_event_priority_orders_leg_completion_before_decision_epoch():
    q = EventQueue()
    t = pd.Timestamp("2016-10-23T00:00:00Z")
    q.push(t, EventType.DECISION_EPOCH, "controller")
    q.push(t, EventType.LEG_COMPLETED, "v1")
    first = q.pop()
    second = q.pop()
    assert first.event_type == EventType.LEG_COMPLETED
    assert second.event_type == EventType.DECISION_EPOCH
    assert EVENT_PRIORITY[EventType.LEG_COMPLETED] < EVENT_PRIORITY[EventType.DECISION_EPOCH]

