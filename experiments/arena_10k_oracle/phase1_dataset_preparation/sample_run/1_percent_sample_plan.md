# Phase 1: 1% Sample Validation Plan

## Overview
Run the complete Phase 1 pipeline with 100 prompts (1% of 10,000) to validate:
1. All scripts execute without errors
2. Teacher forcing implementation works correctly
3. Data formats are consistent
4. Cost and time estimates for full run

## Step-by-Step Execution Plan

### 1. Data Preparation (01_prepare_data.py)
- **Input**: Raw Arena dataset
- **Output**: 100 sample prompts
- **Validation**: 
  - ✓ All prompts have required fields
  - ✓ No duplicate prompt_ids
  - ✓ Proper JSONL format

### 2. Generate P0 Responses (02a_generate_p0_responses.py)
- **Model**: llama4-scout-instruct-basic
- **Expected**: 100 responses
- **Cost**: ~$0.02 (100 calls @ $0.20/1k)
- **Time**: ~2-3 minutes
- **Validation**:
  - ✓ All prompts get responses
  - ✓ Log probabilities included
  - ✓ No empty responses

### 3. Generate Target Responses (02b_generate_target_responses.py)
- **Models**: 
  - pi_clone: llama4-scout-instruct-basic (same as P0)
  - pi_cot: llama4-scout-instruct-basic (with CoT prompt)
  - pi_bigger_model: llama4-maverick-instruct-basic  
  - pi_bad: llama4-scout-instruct-basic (with bad prompt)
- **Expected**: 400 responses (100 × 4 policies)
- **Cost**: ~$0.08 (400 calls @ $0.20/1k)
- **Time**: ~5-8 minutes
- **Validation**:
  - ✓ All policies generate responses
  - ✓ Response diversity across policies

### 4. Compute Teacher Forcing (02c_compute_target_logprobs.py) 🔑 CRITICAL
- **Purpose**: Test robust teacher forcing implementation
- **Expected**: 400 log prob computations
- **Cost**: ~$0.12 (400 calls @ $0.30/1k)
- **Time**: ~10-15 minutes
- **Critical Validation**:
  - ✓ NO 0.0 values for non-empty responses
  - ✓ All three methods tested (token_counting, echo_based, continuation)
  - ✓ Reasonable log prob ranges (-50 to 0)
  - ✓ Method usage statistics logged

### 5. Generate Oracle Labels (03_generate_oracle_labels.py)
- **Model**: gpt-4o
- **Expected**: 100 oracle labels
- **Cost**: ~$0.25 (100 calls @ $2.50/1k)
- **Time**: ~3-5 minutes
- **Validation**:
  - ✓ Labels in valid range
  - ✓ Reasonable distribution

### 6. Compute Judge Scores (04a-04e)
- **Models**: Various judges with uncertainty
- **Expected**: 400 judge scores (100 × 4 responses)
- **Cost**: ~$0.08 (400 calls @ $0.20/1k)
- **Time**: ~8-10 minutes
- **Validation**:
  - ✓ All responses scored
  - ✓ Uncertainty estimates included
  - ✓ Scores properly calibrated

## Total Estimates

### For 1% Sample (100 prompts):
- **Total API Calls**: ~1,200
- **Total Cost**: ~$0.60
- **Total Time**: ~30-45 minutes

### Projected for Full Run (10,000 prompts):
- **Total API Calls**: ~120,000
- **Total Cost**: ~$60
- **Total Time**: ~50-75 hours
- **Recommendation**: Run in parallel batches overnight/weekend

## Validation Checklist

### Pre-Run
- [ ] Verify Fireworks API key and credits
- [ ] Check OpenAI API key for oracle labeling
- [ ] Create sample_1pct directory
- [ ] Set up cost monitoring/alerts

### During Run
- [ ] Monitor for API rate limits
- [ ] Check teacher forcing method distribution
- [ ] Verify no 0.0 log probs for non-empty responses
- [ ] Track checkpoint progress

### Post-Run
- [ ] All output files generated
- [ ] Teacher forcing stats show all methods used
- [ ] No data quality issues flagged
- [ ] Cost tracking matches estimates

## Go/No-Go Criteria

**GO** if all of the following are true:
1. ✅ Zero suspicious 0.0 log probabilities in teacher forcing
2. ✅ All scripts complete without errors
3. ✅ Output data passes validation checks
4. ✅ Costs align with estimates (within 20%)

**NO-GO** if any of:
1. ❌ Teacher forcing returns 0.0 for non-empty responses
2. ❌ Any script fails or produces invalid data
3. ❌ Costs exceed estimates by >50%
4. ❌ API rate limiting causes failures

## Next Steps After Successful Sample Run

1. **Review validation report** in detail
2. **Schedule full run** during low-usage period
3. **Set up monitoring** for overnight execution
4. **Prepare Phase 2** analysis scripts
5. **Document any adjustments** needed for full run