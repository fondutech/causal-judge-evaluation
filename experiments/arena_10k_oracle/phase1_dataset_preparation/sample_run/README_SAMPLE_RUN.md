# Phase 1: 1% Sample Run Guide

## Quick Start

Run the entire 1% sample pipeline with:

```bash
# Run the sample pipeline
./run_sample.sh

# Or run with monitoring in another terminal
python monitor_sample_run.py
```

## What This Does

1. **Samples 100 prompts** (1% of 10,000) from the full dataset
2. **Runs all Phase 1 steps** with the sample
3. **Validates teacher forcing** implementation 
4. **Estimates costs and time** for the full run
5. **Checks for data quality issues**

## Expected Results

### Time: ~30-45 minutes total
- Step 1 (Prepare data): <1 minute
- Step 2a (P0 responses): 2-3 minutes  
- Step 2b (Target responses): 5-8 minutes
- Step 2c (Teacher forcing): 10-15 minutes ⚠️ CRITICAL
- Step 3 (Oracle labels): 3-5 minutes
- Step 4 (Judge scores): 8-10 minutes

### Cost: ~$0.50 total
- Fireworks API: ~$0.17 (700 calls)
- OpenAI API: ~$0.25 (100 calls)
- Buffer: ~$0.08

### Output Files
All outputs saved to `../data/sample_1pct/`:
- `arena_questions_base_sample.jsonl` - 100 prompts
- `p0_replies_sample.jsonl` - P0 responses  
- `target_responses_sample.jsonl` - 300 target responses
- `p0_with_target_logps_sample.jsonl` - Teacher forcing results
- `oracle_labels_sample.jsonl` - Oracle labels
- `*_scored_sample.jsonl` - Judge scores

## Validation Checklist

### ✅ Pre-Run
- [ ] Fireworks API key set (`FIREWORKS_API_KEY`)
- [ ] OpenAI API key set (`OPENAI_API_KEY`) 
- [ ] Base dataset exists (`01_prepare_data.py` run on full data)

### ✅ During Run
- [ ] Monitor script shows progress
- [ ] No API rate limit errors
- [ ] Teacher forcing uses all 3 methods

### ✅ Post-Run (CRITICAL)
- [ ] **NO 0.0 log probabilities for non-empty responses** ⚠️
- [ ] All output files generated
- [ ] Validation script passes:
  ```bash
  python validate_sample_results.py
  ```

## Troubleshooting

### "Base dataset not found"
Run `python 01_prepare_data.py` first to create the full dataset.

### API errors
- Check API keys are set correctly
- Verify you have credits/quota
- Check rate limits

### Teacher forcing returns 0.0
**STOP!** This indicates the bug is not fixed. Check:
1. Using `RobustTeacherForcing` from `cje.utils`
2. All 3 methods implemented (token_counting, echo_based, continuation)
3. Review the validation output

## Next Steps

### If validation passes ✅
1. Review cost and time estimates
2. Schedule full run (50-75 hours)
3. Set up monitoring/alerts
4. Run full pipeline:
   ```bash
   unset ARENA_SAMPLE_MODE
   ./run_full_pipeline.sh
   ```

### If validation fails ❌
1. Check `validation_report.md` for details
2. Fix identified issues
3. Re-run sample test
4. Do NOT proceed to full run until fixed

## Full Run Estimates (10,000 prompts)

Based on 1% sample:
- **Time**: 50-75 hours (run over weekend)
- **Cost**: ~$50 
- **API Calls**: ~120,000
- **Storage**: ~500MB

## Questions?

Check the detailed plan in `1_percent_sample_plan.md` or the validation report in `sample_1pct/validation_report.md`.