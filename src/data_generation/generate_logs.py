"""Generates synthetic maintenance and failure records to complement sensor data.

These records simulate real-world unstructured text logs that would be
found in an MRO (maintenance, repair, overhaul) system.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from faker import Faker

DATA_DIR = Path(__file__).parent.parent.parent / 'data'
DATA_DIR.mkdir(exist_ok=True)

fake = Faker()

FAILURE_MODES = list({
    'High-pressure turbine wear',
    'Compressor stall',
    'Main bearing failure',
    'Foreign object damage',
    'Overheating',
})

RECORD_TYPES = ['scheduled_inspection', 'unscheduled_repair', 'failure']

DESCRIPTION_TEMPLATES = {
    'scheduled_inspection': [
        "Routine {hours}-hour inspection. No issues found.",
        "Visual check: Blades intact, oil clean.",
    ],
    'unscheduled_repair': [
        "Abnormal {sensor} detected. Diagnosed as {cause}.",
        "Repaired {part} due to {issue}.",
    ],
    'failure': [
        "Catastrophic {failure_mode}. Engine shutdown.",
        "Failure analysis: Root cause {cause}.",
    ],
}


def generate_records_for_engine(
    engine_id: str,
    num_cycles: int,
    failure_cycle: int,
    failure_mode: str,
) -> list[dict]:
    """Create realistic maintenance history for one engine."""
    records = []
    base_ts = datetime(2025, 11, 1)
    num_records = random.randint(5, 20)

    for _ in range(num_records):
        cycle_link = random.randint(1, num_cycles)
        ts = base_ts + timedelta(days=random.randint(0, 365), hours=random.randint(0, 23))

        if cycle_link == failure_cycle:
            rec_type = 'failure'
        else:
            rec_type = random.choice(RECORD_TYPES[:-1] if random.random() < 0.8 else RECORD_TYPES)

        severity = 'critical' if rec_type == 'failure' else random.choice(['low', 'medium', 'high'])

        template = random.choice(DESCRIPTION_TEMPLATES[rec_type])
        description = template.format(
            hours=random.randint(100, 500),
            sensor=random.choice(['vibration', 'temperature']),
            cause=failure_mode.lower(),
            part=random.choice(['blades', 'bearings']),
            issue='wear',
            failure_mode=failure_mode,
        )

        records.append({
            'record_id': f"MNT-{engine_id}-{ts.strftime('%Y-%m-%d')}",
            'engine_id': engine_id,
            'timestamp': ts.isoformat(),
            'type': rec_type,
            'description': description,
            'actions_taken': fake.sentence(nb_words=6),
            'parts_replaced': [fake.word() for _ in range(random.randint(0, 3))],
            'technician': fake.name(),
            'related_failure_mode': failure_mode if rec_type == 'failure' else None,
            'severity': severity,
            'linked_cycles': [cycle_link],
            'downtime_hours': random.randint(10, 500) if rec_type in ['unscheduled_repair', 'failure'] else None,
        })

    return records


if __name__ == '__main__':
    engines = [f'ENG_{i:03d}' for i in range(1, 501)]
    all_records = []

    for engine_id in engines:
        num_cycles = random.randint(100, 300)
        failure_cycle = random.randint(50, num_cycles)
        failure_mode = random.choice(FAILURE_MODES)
        all_records.extend(generate_records_for_engine(engine_id, num_cycles, failure_cycle, failure_mode))

    with open(DATA_DIR / 'synthetic_maintenance_records.json', 'w') as f:
        json.dump(all_records, f, indent=2)

    print(f"Generated {len(all_records)} maintenance records.")