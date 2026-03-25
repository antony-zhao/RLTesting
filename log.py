from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator('logs/dreamer-v3/Assault/2026-03-23-18-15-43/events.out.tfevents.1774311343.antony-B650-AORUS-ELITE-AX.2444701.0')
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