import asyncio

def log_start():
    print("[START] task=test env=openenv model=dummy", flush=True)

def log_step(step, action, reward, done, error=None):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

async def main():
    log_start()
    rewards = []
    steps = 0

    for i in range(3):
        steps += 1
        reward = 0.1 * i
        rewards.append(reward)
        log_step(steps, "test_action", reward, False)

    log_end(True, steps, rewards)

if __name__ == "__main__":
    asyncio.run(main())
