<p class="kicker">Compute · Notes</p>

# Open questions

Things I'm still working through — hypotheses, not conclusions.

## Will the benchmark matter, or only regionally?

Hypothesis (Oct 13, 2026): The race to become *the* benchmark for compute — Ornn's OCPI via ICE, Silicon Data via CME — may be overstated. A single broad-based benchmark may not provide as much value as assumed.

If the benchmark matters at all, it may only matter regionally — because compute prices vary by region, and a broad average washes out the differences that actually drive buying and selling decisions.

Still to explore:

- What would prove this wrong: liquidity concentrating around one broad benchmark, with regional differences trading as a small basis on top of it.
- What would prove this right: benchmarks fragmenting by region / power market / latency, with most hedging done locally or bilaterally.
- Whether physical settlement and bilateral contracts bypass the need for a broad benchmark entirely.

## When a buyer buys compute on these platforms, how do they actually turn it into usable compute?

Question to answer (Oct 13, 2026): When buyers buy compute on these platforms — compute marketplaces / sandbox companies — what is the process of turning that purchase into compute they can actually use?

Still to explore:

- What does the buyer actually get — an API key, SSH access, a container, a cluster?
- How do provisioning, scheduling, networking, and storage work between paying and running a job?
- How does that differ for spot vs. reserved vs. futures-settled / physically-settled capacity?

## Side thought: compute as a financial stack — track marginal output per GPU-hour

Side thought (Oct 13, 2026): Compute is valuable as it is stored. The output of electricity / oil is only as valuable as the outputs that can be created with it — same for compute.

Think of compute as a financial stack: as a buyer, can you track the marginal output per GPU-hour, and easily compare whether you can downgrade along the stack?

Still to explore:

- What counts as marginal output per GPU-hour — tokens, training progress, completed jobs?
- What does downgrading along the stack mean in practice — cheaper GPU, smaller model, lower precision — and when does it preserve output per dollar?
- If buyers could see marginal output clearly, would price be set by output rather than by GPU-hour?

---

*Related: [Why the benchmark](why-the-benchmark.md) — Gupta / Mizuho on the benchmark as the real innovation · [Back to overview](index.md)*
