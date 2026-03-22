from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator('logs/dreamer_crl/2026-03-21-00-50-14/events.out.tfevents.1774075814.antony-B650-AORUS-ELITE-AX.155699.0')
ea.Reload()
for tag in sorted(ea.Tags()['scalars']):
    events = ea.Scalars(tag)
    n = len(events)
    indices = [0, 1, 2, n//4, n//2, 3*n//4, n-3, n-2, n-1]
    print(f'\n--- {tag} ({n} entries) ---')
    for i in indices:
        if 0 <= i < n:
            e = events[i]
            print(f'  step={e.step:>8d} value={e.value:.6f}')