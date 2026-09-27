import { ethers } from "ethers";
import { readFile } from "node:fs/promises";
import { spawn, execFileSync } from "node:child_process";
import { randomBytes } from "node:crypto";

const dep = JSON.parse(await readFile("./deployment.json", "utf8"));
const pk = process.env.PURRCAT_PRIVATE_KEY?.trim();
if (!pk) throw new Error("PURRCAT_PRIVATE_KEY is missing");
if (Number(dep.chainId) !== 999) throw new Error(`Unexpected chainId ${dep.chainId}`);

const rpc = (Array.isArray(dep.rpcs) && dep.rpcs.length ? dep.rpcs : [dep.rpc]).find(Boolean);
if (!rpc) throw new Error("No RPC configured");

const provider = new ethers.JsonRpcProvider(rpc, 999, { staticNetwork: true });
const wallet = new ethers.Wallet(pk, provider);
const me = wallet.address;
const nft = new ethers.Contract(dep.address, dep.abi, provider);
const signedNft = nft.connect(wallet);
const coder = ethers.AbiCoder.defaultAbiCoder();
const { prepare, digestHex, nonceHex } = await import("./keccak_core.js");

const TWO256 = 1n << 256n;
const ANCHOR_BACK = 3;
const ANCHOR_REFRESH = 150;
const ANCHOR_MAX_SEND = 240;
const MAX_WINS = Number(process.env.PURRCAT_MAX_WINS || 1);
const GPU_BIN = "./purrcat_cuda_worker";

function gpuIds() {
  const raw = process.env.PURRCAT_GPUS;
  if (raw) return raw.split(",").map(s => Number(s.trim())).filter(Number.isInteger);
  const n = Number(process.env.PURRCAT_GPU_COUNT || 0);
  if (n > 0) return Array.from({length:n}, (_,i)=>i);
  return null;
}
function detectGpuIds() {
  if (gpuIds()) return gpuIds();
  const out = execFileSync("nvidia-smi", ["--query-gpu=index,name", "--format=csv,noheader"], {encoding:"utf8"});
  return out.trim().split(/\n+/).filter(Boolean).map(x => Number(x.split(",")[0].trim()));
}
const ids = detectGpuIds();
if (!ids.length) throw new Error("No NVIDIA GPUs detected.");

console.log("============================================");
console.log("      PURRCAT MULTI-GPU 5090 MINER");
console.log("============================================");
console.log("Wallet :", me);
console.log("GPUs   :", ids.join(", "));
console.log("Count  :", ids.length);
console.log("============================================");

let workers = [];
let workerTotals = new Map(ids.map(i => [i, 0]));
let job = null;
let jobId = 0;
let winner = false;
let wins = 0;
let lastTotal = 0;
let lastAt = Date.now();

function stopWorkers() {
  for (const w of workers) {
    try { w.p.stdin.write("STOP\n"); } catch {}
    setTimeout(() => { try { w.p.kill("SIGTERM"); } catch {} }, 250);
  }
}
function launchWorkers() {
  workers = ids.map((gpu, wid) => {
    const p = spawn(GPU_BIN, [String(gpu), String(ids.length), String(wid)], {stdio:["pipe","pipe","inherit"]});
    p.stdout.setEncoding("utf8");
    let buf = "";
    p.stdout.on("data", chunk => {
      buf += chunk;
      let i;
      while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i).trim();
        buf = buf.slice(i+1);
        if (!line) continue;
        const parts = line.split(/\s+/);
        if (parts[0] === "STAT" && parts.length >= 3) {
          const jid = Number(parts[1]);
          if (job && jid === job.jobId) workerTotals.set(gpu, Number(parts[2]));
        } else if (parts[0] === "FOUND" && parts.length >= 4) {
          const jid = Number(parts[1]);
          const hi = Number(parts[2]) >>> 0;
          const ctr = Number(parts[3]) >>> 0;
          if (job && jid === job.jobId && !winner) void handleFound(jid, hi, ctr);
        }
      }
    });
    p.on("exit", (code, sig) => { if (!winner) console.error(`[GPU ${gpu}] exited code=${code} signal=${sig}`); });
    return {gpu, wid, p};
  });
}
function broadcastJob(j) {
  const line = ["JOB", j.jobId, j.hiBase >>> 0, j.targetHi >>> 0, j.targetLo >>> 0, ...j.base].join(" ") + "\n";
  workerTotals = new Map(ids.map(i => [i, 0]));
  for (const w of workers) { try { w.p.stdin.write(line); } catch (e) { console.error(`[GPU ${w.gpu}] job send failed: ${e.message}`); } }
}
async function getState() {
  const s = await nft.state(me);
  const start = await nft.startTime();
  return { minted:Number(s.minted), prev:s.prev, target:s.yourTarget, block:Number(s.blockNumber), started:start>0n };
}
function diffBits(target) {
  if (!target || target <= 0n) return 256;
  const len = target.toString(2).length;
  const top = Number(target >> BigInt(Math.max(0,len-53)));
  return 256 - (Math.log2(top) + Math.max(0,len-53));
}
function fmtRate(h) {
  if (h >= 1e9) return (h/1e9).toFixed(2)+" GH/s";
  if (h >= 1e6) return (h/1e6).toFixed(2)+" MH/s";
  return (h/1e3).toFixed(1)+" kH/s";
}
function fmtEta(sec) {
  if (!Number.isFinite(sec)) return "--";
  if (sec < 60) return Math.max(1,Math.round(sec))+"s";
  if (sec < 3600) return Math.round(sec/60)+"m";
  if (sec < 86400) return (sec/3600).toFixed(1)+"h";
  return (sec/86400).toFixed(1)+"d";
}
async function feeOverrides() {
  const f = await provider.getFeeData();
  const floor = ethers.parseUnits(String(dep.priorityGwei ?? 0.1), "gwei");
  const prio = f.maxPriorityFeePerGas && f.maxPriorityFeePerGas > floor ? f.maxPriorityFeePerGas : floor;
  if (f.maxFeePerGas == null) return {gasPrice:(f.gasPrice||0n)*2n+prio};
  return {maxPriorityFeePerGas:prio,maxFeePerGas:f.maxFeePerGas*2n+prio};
}
async function submit(j, hi, ctr) {
  if (winner) return;
  winner = true;
  stopWorkers();
  try {
    const nonce = nonceHex(j.nonceHigh, hi, ctr);
    const digest = BigInt(digestHex(j.prep, hi, ctr));
    if (digest >= j.target) throw new Error("Candidate failed full-digest check");
    const head = await provider.getBlockNumber();
    if (head - j.anchor > ANCHOR_MAX_SEND) throw new Error("Candidate became stale");
    const minted = Number(await nft.totalMinted());
    const price = await nft.priceOf(minted + 1);
    console.log("[FOUND] Valid proof.");
    console.log("[CHECK] Static call...");
    await nft.mine.staticCall(j.anchor, nonce, price, {value:price, from:me});
    console.log("[SEND] mine()...");
    const tx = await signedNft.mine(j.anchor, nonce, price, {value:price,gasLimit:450000n,...(await feeOverrides())});
    console.log("[TX]", tx.hash);
    const receipt = await tx.wait();
    let tokenId = "?";
    for (const log of receipt.logs) {
      try { const p = nft.interface.parseLog(log); if (p?.name === "Mined") { tokenId = p.args.id.toString(); break; } } catch {}
    }
    wins++;
    console.log("============================================");
    console.log("PURRCAT MINT SUCCESS");
    console.log("PurrCat #"+tokenId);
    console.log("TX: "+tx.hash);
    console.log("============================================");
  } catch (e) {
    console.error("[SUBMIT ERROR]", e.shortMessage || e.message || e);
    winner = false;
    if (job) broadcastJob(job);
  }
}
async function handleFound(jid, hi, ctr) { if (!job || jid !== job.jobId || winner) return; await submit(job, hi, ctr); }
async function newJob(st) {
  const anchor = st.block - ANCHOR_BACK;
  const block = await provider.getBlock(anchor);
  if (!block?.hash) return;
  const challenge = ethers.keccak256(coder.encode(["bytes32","bytes32"],[block.hash, st.prev]));
  const nonceHigh = ethers.hexlify(randomBytes(32));
  const spec = {domain:dep.domain,chainId:dep.chainId,contract:dep.address,anchorHash:challenge,miner:me,nonceHigh};
  const prep = prepare(spec);
  const base = new Uint32Array(50);
  for (let i=0;i<50;i++) base[i] = (prep.mid[i] ^ (i<34 ? prep.tailLanes[i] : 0)) >>> 0;
  const rnd = randomBytes(4);
  const hiBase = (((rnd[0]<<24)|(rnd[1]<<16)|(rnd[2]<<8)|rnd[3])>>>0);
  job = {...spec,jobId:++jobId,anchor,target:st.target,prev:st.prev,prep,base:Array.from(base),hiBase,targetHi:Number(st.target >> 224n),targetLo:Number((st.target >> 192n)&0xffffffffn)};
  winner = false;
  broadcastJob(job);
  console.log(`[JOB] #${job.jobId} block=${anchor} difficulty=${diffBits(st.target).toFixed(1)} bits`);
}
process.on("SIGINT",()=>{stopWorkers();process.exit(0);});
process.on("SIGTERM",()=>{stopWorkers();process.exit(0);});
launchWorkers();
console.log("[WALLET]", ethers.formatEther(await provider.getBalance(me)), "HYPE");
setInterval(()=>{
  const total=[...workerTotals.values()].reduce((a,b)=>a+b,0);
  const now=Date.now(); const dt=(now-lastAt)/1000;
  const rate=dt>0?(total-lastTotal)/dt:0;
  lastTotal=total; lastAt=now;
  let eta="--"; if(job&&rate>0) eta=fmtEta(Number(TWO256/job.target)/rate);
  console.log(`[STATUS] HASH ${fmtRate(rate)} | AVG ETA ${eta} | DIFF ${job?diffBits(job.target).toFixed(1):"--"} bits | TOTAL ${total.toLocaleString()}`);
},5000);
for (;;) {
  if (wins >= MAX_WINS) break;
  try {
    const st=await getState();
    if (st.minted>=3333) { console.log("[STOP] 3,333 minted."); break; }
    if (!st.started) { console.log("[WAIT] Hunt not open."); }
    else {
      const stale=!job||job.prev!==st.prev||job.target!==st.target||st.block-job.anchor>ANCHOR_REFRESH;
      if (stale) { if(job) stopWorkers(); await new Promise(r=>setTimeout(r,150)); await newJob(st); if (workers.some(w=>w.p.exitCode!==null)) launchWorkers(); }
    }
  } catch(e) { console.error("[RPC]", e.message||e); }
  await new Promise(r=>setTimeout(r,1500));
}
stopWorkers();
