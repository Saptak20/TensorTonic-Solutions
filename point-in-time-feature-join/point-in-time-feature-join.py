from collections import defaultdict
import bisect

def point_in_time_join(events: list, feature_rows: list, ttl: int) -> list:
    grouped = defaultdict(list)
    for row in feature_rows:
        grouped[row["entity_id"]].append((row["timestamp"], row["value"]))

    times = {}
    values = {}
    for entity, rows in grouped.items():
        rows.sort(key=lambda r: r[0])
        times[entity] = [r[0] for r in rows]
        values[entity] = [r[1] for r in rows]

    result = []
    for event in events:
        entity = event["entity_id"]
        ts = event["timestamp"]
        value = None
        if entity in times:
            idx = bisect.bisect_right(times[entity], ts) - 1
            if idx >= 0 and ts - times[entity][idx] <= ttl:
                value = values[entity][idx]
        result.append({"entity_id": entity, "timestamp": ts, "value": value})
    return result