# Arena 10K Phase 1: Final Execution Checklist

## 🚦 Pre-Launch Checklist

### 1. Environment Setup
```bash
# Check everything is ready
python preflight_check.py

# Should show all green checkmarks for:
# ✅ Python 3.8+
# ✅ FIREWORKS_API_KEY set
# ✅ OPENAI_API_KEY set  
# ✅ CJE package importable
# ✅ RobustTeacherForcing available
# ✅ Base dataset exists
```

### 2. Cost Estimation
```bash
# Review API costs
python estimate_costs.py

# Expected for 1% sample:
# - Total cost: ~$0.50
# - API calls: ~1,200
# 
# Expected for full run:
# - Total cost: ~$50
# - API calls: ~120,000
```

## 🧪 Sample Run (1% - 100 prompts)

### Step 1: Quick Test
```bash
# Verify sample configuration works
python quick_sample_test.py
```

### Step 2: Run Sample Pipeline
```bash
# Terminal 1: Run pipeline
./run_sample.sh

# Terminal 2: Monitor progress (optional)
python monitor_sample_run.py
```

### Step 3: Validate Results ⚠️ CRITICAL
```bash
# Run validation
python validate_sample_results.py

# MUST see:
# ✅ Teacher Forcing: PASS
# ✅ Data Consistency: PASS
# ✅ No suspicious 0.0 values
```

### Step 4: Review Reports
- Check `data/sample_1pct/validation_report.md`
- Verify teacher forcing stats show all 3 methods used
- Confirm no 0.0 log probs for non-empty responses

## 🚀 Full Run (10,000 prompts)

### Only proceed if sample validation passed!

### Step 1: Final Preflight
```bash
# Check for full run
python preflight_check.py --full
```

### Step 2: Schedule Run
```bash
# Unset sample mode
unset ARENA_SAMPLE_MODE

# Run full pipeline (50-75 hours)
nohup ./run_full_pipeline.sh > phase1_full.log 2>&1 &

# Get process ID
echo $! > phase1.pid
```

### Step 3: Monitor Progress
```bash
# Check log file
tail -f phase1_full.log

# Monitor output files
watch -n 60 'ls -la ../data/*.jsonl | tail -10'

# Check for errors
grep -i error phase1_full.log
```

## 🔍 Critical Validation Points

### During Teacher Forcing (Step 2c)
Watch for:
- "Warning: Got 0.0 for non-empty response" ❌
- "Method: token_counting|echo_based|continuation" ✅
- Method distribution should show all 3 being used

### After Each Step
Verify:
1. Output file exists and growing
2. No repeated errors in logs
3. Checkpoint files being updated

### Final Validation
```bash
# Count entries in each file
for f in ../data/*.jsonl; do
  echo "$f: $(wc -l < $f) entries"
done

# Should see:
# arena_questions_base.jsonl: 10000 entries
# p0_replies.jsonl: 10000 entries
# target_responses.jsonl: 30000 entries
# p0_with_target_logps.jsonl: 10000 entries
# oracle_labels.jsonl: 10000 entries
# *_scored.jsonl files with appropriate counts
```

## ⚠️ Troubleshooting

### API Rate Limits
- Fireworks: Add delays between batches
- OpenAI: Reduce batch size for oracle labeling

### Memory Issues
- Reduce batch sizes in scripts
- Use checkpoint recovery to resume

### Teacher Forcing Failures
- Check API keys and credits
- Verify model names are correct
- Review error logs for specific failures

## 📊 Success Criteria

### Sample Run Success:
- [x] All scripts complete without errors
- [x] No 0.0 log probs for non-empty responses  
- [x] Validation script shows "READY FOR FULL RUN"
- [x] Costs within 20% of estimates

### Full Run Success:
- [ ] All 10,000 prompts processed
- [ ] Teacher forcing shows <1% failure rate
- [ ] No suspicious 0.0 values in final data
- [ ] Total cost under $75 (50% buffer)
- [ ] Ready for Phase 2 analysis

## 🎯 Next Steps After Success

1. **Backup the data**:
   ```bash
   tar -czf arena_10k_phase1_backup.tar.gz ../data/*.jsonl
   ```

2. **Generate summary statistics**:
   ```bash
   python ../phase2_cje_ablations/analyze_data_quality.py
   ```

3. **Proceed to Phase 2**: Run CJE experiments with the prepared data

---

Remember: The 1% sample run is your safety net. Don't skip it!