import sys, time, importlib.util, uuid, json, random, itertools, multiprocessing as mp
def load(path):
    from agent.loader import load_agent          # Kaggle's entry rule (last callable), not module.agent
    return load_agent(path)
def play(args):
    a, b, seed = args
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    t = time.time()
    env.run([load(a), load(b)])
    st = env.steps[-1]
    return a, b, seed, st[0].reward, st[1].reward, st[0].status, st[1].status, time.time() - t
if __name__ == "__main__":
    agents = sys.argv[1].split(","); n = int(sys.argv[2]); procs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    jobs = []
    for s in range(n):
        seed = 1000 + s
        for a, b in itertools.combinations(agents, 2):
            jobs += [(a, b, seed), (b, a, seed)]
    res = {}
    with mp.Pool(procs) as p:
        for r in p.imap_unordered(play, jobs):
            a, b, seed, ra, rb, sa, sb, dt = r
            print(f"{a.split('/')[-1]:>14} {ra!s:>8} vs {b.split('/')[-1]:<14} {rb!s:>8} seed={seed} {sa}/{sb} {dt:.0f}s", flush=True)
            for x, rx, ry in ((a, ra, rb), (b, rb, ra)):
                w = res.setdefault(x, [0, 0, 0, 0.0])
                rx = rx or 0; ry = ry or 0
                w[0] += rx > ry; w[1] += rx == ry; w[2] += rx < ry; w[3] += rx
    for k, (w, d, l, m) in sorted(res.items(), key=lambda kv: -kv[1][0]):
        print(f"{k.split('/')[-1]:>14}  W{w} D{d} L{l}  winrate={(w+0.5*d)/max(1,w+d+l):.3f}  avg_money={m/max(1,w+d+l):.0f}")
