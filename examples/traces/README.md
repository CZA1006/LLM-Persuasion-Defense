# Curated Trace Examples

This directory retains the two G5 (Hybrid) case studies discussed in Appendix A of the final report. They are compact extracts from `trace_20251217_020353_dsk_G5_hybrid_smart_t1to8_k1.jsonl`, not synthetic reconstructions.

- `christian_worship_g5.jsonl`: five turns, ending with `hit_o_false=true` for the target false value `Islam`.
- `delta_goodrem_g5.jsonl`: two turns, ending with `hit_o_false=true` for the target false value `India`.

Each JSONL record preserves the original prompt, target response, strategy, diagnosis, configuration, and timestamp. Full raw experiment traces are intentionally excluded because they are large generated artifacts; reruns write them to `traces/`.

To curate another contiguous case trajectory:

```bash
python tools/curate_trace_example.py SOURCE.jsonl OUTPUT.jsonl \
  --subject "Subject Name" \
  --start-ts 2025-12-17T00:00:00 \
  --end-ts 2025-12-17T00:05:00
```
