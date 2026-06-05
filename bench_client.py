import sys, json, time, threading, urllib.request
PORT=int(sys.argv[1]) if len(sys.argv)>1 else 8002
TAG=sys.argv[2] if len(sys.argv)>2 else "run"
URL=f"http://127.0.0.1:{PORT}/v1/completions"
PROMPT="Write a detailed paragraph about the history of computing, covering at least five major milestones."
MAXTOK=256
def one():
    body=json.dumps({"model":"step3p7","prompt":PROMPT,"max_tokens":MAXTOK,"temperature":0.7,"ignore_eos":True}).encode()
    t0=time.time()
    r=urllib.request.urlopen(urllib.request.Request(URL,body,{"Content-Type":"application/json"}),timeout=300)
    d=json.load(r); dt=time.time()-t0
    n=d.get("usage",{}).get("completion_tokens",MAXTOK)
    return n,dt
def bench(conc,reps=2):
    rates=[]
    for _ in range(reps):
        res=[None]*conc; ths=[]
        t0=time.time()
        def w(i):
            res[i]=one()
        for i in range(conc):
            th=threading.Thread(target=w,args=(i,)); th.start(); ths.append(th)
        for th in ths: th.join()
        wall=time.time()-t0
        toks=sum(n for n,_ in res)
        rates.append(toks/wall)
    return sum(rates)/len(rates)
# warmup
one()
out={}
for c in [1,2,4]:
    r=bench(c)
    out[c]=round(r,2)
    print(f"conc={c}: {r:.2f} tok/s aggregate")
json.dump({"tag":TAG,"port":PORT,"agg_tok_s":out}, open(f"/home/hikari/bench/results/step37_{TAG}.bench.json","w"), indent=2)
print("saved", f"step37_{TAG}.bench.json")
