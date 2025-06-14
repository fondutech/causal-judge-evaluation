Trajectory Methods Guide
========================

CJE supports multi-turn conversation analysis through trajectory-based evaluation, allowing you to analyze and improve conversational AI systems step-by-step.

Overview
--------

**Trajectory evaluation** analyzes multi-turn conversations as sequences of interactive steps, where each step consists of a user input and assistant response. This enables:

- **Turn-level analysis**: Understand which conversation turns perform better
- **Context effects**: Measure how conversation history impacts quality  
- **Interactive optimization**: Improve multi-turn conversation strategies
- **Dialogue policy evaluation**: Compare different conversation management approaches

What is a Trajectory?
---------------------

A trajectory represents a complete multi-turn conversation:

.. code-block:: json

   {
     "uid": "conversation_001",
     "steps": [
       {
         "context": "Hello, I need help with my order",
         "response": "I'd be happy to help! Can you provide your order number?",
         "step_reward": 0.8
       },
       {
         "context": "My order number is #12345",
         "response": "I found your order. It's currently being processed and will ship tomorrow.",
         "step_reward": 0.9
       }
     ],
     "y_true": 0.85,  // Overall conversation quality
     "meta": {
       "conversation_type": "customer_support",
       "user_satisfaction": "high"
     }
   }

**Key Components:**

- **Steps**: Individual turns in the conversation
- **Context**: User input (may include conversation history)
- **Response**: Assistant output for this turn  
- **Step rewards**: Quality scores for individual turns
- **Overall reward**: Final conversation outcome

Configuration
-------------

Basic Trajectory Setup
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Dataset configuration
   dataset:
     name: "./data/conversations.jsonl"
     format: "trajectory"          # Enable trajectory mode
     
   # Trajectory-specific settings
   trajectory:
     max_turns: 10                 # Limit conversation length
     context_window: 3             # Include last 3 turns as context
     step_aggregation: "mean"      # How to combine step rewards
     
   # Estimator configuration
   estimator:
     name: "DRCPO"                 # Doubly-robust (recommended)
     k: 5                          # Cross-validation folds
     clip: 20.0                    # Importance weight clipping
     # Trajectory mode automatically enabled

Data Formats
~~~~~~~~~~~~

**JSONL Format** (recommended):

.. code-block:: json

   {
     "uid": "conv_001",
     "steps": [
       {"context": "Hello", "response": "Hi! How can I help?"},
       {"context": "I have a question", "response": "What's your question?"}
     ],
     "y_true": 0.8
   }

**CSV Format** (flattened):

.. code-block:: csv

   conversation_id,turn,context,response,step_reward,final_reward
   conv_001,1,"Hello","Hi! How can I help?",0.7,0.8
   conv_001,2,"I have a question","What's your question?",0.9,0.8

Analysis Types
--------------

1. Turn-Level Analysis
~~~~~~~~~~~~~~~~~~~~~~

Analyze performance at each conversation turn:

.. code-block:: python

   from cje.analysis import TrajectoryAnalyzer

   analyzer = TrajectoryAnalyzer(dataset)
   
   # Performance by turn position
   turn_performance = analyzer.analyze_by_turn()
   print(f"Turn 1 avg reward: {turn_performance[1]:.3f}")
   print(f"Turn 3 avg reward: {turn_performance[3]:.3f}")

**Use Cases:**

- Identify where conversations typically break down
- Optimize early-turn greeting and engagement strategies
- Understand conversation length effects

2. Context Window Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

Study how conversation history affects current responses:

.. code-block:: python

   # Compare different context window sizes
   results = analyzer.compare_context_windows([1, 3, 5, 10])
   
   # Context utilization analysis
   context_effects = analyzer.analyze_context_utilization()

**Insights:**

- Optimal context window length for your domain
- Whether models effectively use conversation history
- Memory vs. performance tradeoffs

3. Dialogue Strategy Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare different conversation management approaches:

.. code-block:: yaml

   # Target policies (what we want to evaluate)
   target_policies:
     - name: "short_responses"
       provider: "openai"
       model_name: "gpt-4o-mini"
       temperature: 0.7
       mc_samples: 5               # Monte Carlo samples per context
       system_prompt: "Give brief, concise responses"
       max_tokens: 50
       
     - name: "detailed_responses"  
       provider: "openai"
       model_name: "gpt-4o-mini"
       temperature: 0.7
       mc_samples: 5               # Monte Carlo samples per context
       system_prompt: "Provide detailed, helpful responses"
       max_tokens: 200
       
     - name: "question_focused"
       provider: "openai"
       model_name: "gpt-4o-mini"
       temperature: 0.7
       mc_samples: 5               # Monte Carlo samples per context
       system_prompt: "Always ask clarifying questions"

**Evaluation Dimensions:**

- User satisfaction over conversation length
- Task completion rates  
- Conversation efficiency metrics

Advanced Features
-----------------

Step-Level Importance Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CJE can compute importance weights at the step level for fine-grained analysis:

.. code-block:: python

   # Enable step-level weights
   estimator = MultiDRCPOEstimator(
       step_level_weights=True,
       context_aggregation="hierarchical"
   )

**Benefits:**

- Higher precision for multi-turn analysis
- Better handling of context dependencies
- Reduced variance in long conversations

Hierarchical Evaluation
~~~~~~~~~~~~~~~~~~~~~~~

Combine step-level and conversation-level rewards:

.. code-block:: yaml

   # Trajectory configuration
   trajectory:
     evaluation_mode: "hierarchical"
     step_weight: 0.3              # Weight for step-level rewards
     conversation_weight: 0.7      # Weight for overall outcome

Conversation State Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Model conversation state evolution:

.. code-block:: python

   # Track conversation state
   analyzer.track_conversation_states([
       "greeting", "information_gathering", "problem_solving", "resolution"
   ])
   
   # Analyze state transition performance
   transition_analysis = analyzer.analyze_state_transitions()

Common Patterns
---------------

Customer Support Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Analyze support conversation quality
   support_analyzer = TrajectoryAnalyzer(
       dataset="customer_support_logs.jsonl",
       success_metric="issue_resolved",
       efficiency_metric="turns_to_resolution"
   )
   
   # Key metrics for support
   results = support_analyzer.analyze_support_quality()

Educational Tutoring
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Tutoring conversation analysis
   tutor_analyzer = TrajectoryAnalyzer(
       dataset="tutoring_sessions.jsonl", 
       learning_objectives=["concept_understanding", "engagement", "retention"]
   )
   
   # Learning progression analysis
   learning_curves = tutor_analyzer.analyze_learning_progression()

Sales Conversations
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Sales conversation optimization
   sales_analyzer = TrajectoryAnalyzer(
       dataset="sales_calls.jsonl",
       outcome_metrics=["conversion", "customer_interest", "objection_handling"]
   )
   
   # Conversion funnel analysis
   funnel_performance = sales_analyzer.analyze_conversion_funnel()

Best Practices
--------------

Data Collection
~~~~~~~~~~~~~~~

**Conversation Boundaries:**

- Clearly define conversation start/end points
- Handle conversation resumption appropriately
- Maintain consistent user identity across turns

**Context Management:**

- Include relevant conversation history in each turn
- Balance context length vs. computational efficiency
- Handle long conversations with sliding windows

**Quality Annotation:**

- Collect both step-level and conversation-level labels when possible
- Use consistent annotation guidelines across annotators
- Consider multiple quality dimensions (helpfulness, accuracy, engagement)

Evaluation Design
~~~~~~~~~~~~~~~~~

**Turn Sampling:**

- Ensure representative sampling across conversation lengths
- Balance early vs. late turn performance
- Account for different conversation types

**Baseline Comparison:**

- Compare against turn-independent baselines
- Include human performance benchmarks
- Test across different conversation scenarios

**Statistical Considerations:**

- Account for conversation-level clustering
- Use appropriate confidence intervals for hierarchical data
- Consider multiple testing corrections for turn-wise analysis

Troubleshooting
---------------

Low Performance on Later Turns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**

- Performance degrades after turn 3-5
- Confidence intervals widen for later turns
- Context seems ignored in responses

**Solutions:**

- Increase context window size
- Use models with longer context limits
- Implement conversation summarization
- Add explicit state tracking

Inconsistent Conversation Quality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**

- High variance in conversation-level outcomes
- Step-level rewards don't predict overall success
- User satisfaction doesn't correlate with model scores

**Solutions:**

- Improve conversation-level reward modeling
- Add conversation flow coherence metrics
- Use hierarchical evaluation approaches
- Collect more nuanced quality annotations

Memory and Performance Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**

- Slow processing of long conversations
- Memory errors with large context windows
- Timeout errors in evaluation

**Solutions:**

- Implement conversation chunking
- Use sliding context windows
- Optimize model inference batching
- Consider conversation-level sampling

Integration Examples
--------------------

Real-Time Conversation Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class ConversationOptimizer:
       def __init__(self, trajectory_analyzer):
           self.analyzer = trajectory_analyzer
           
       def optimize_next_response(self, conversation_history):
           # Analyze current conversation state
           current_state = self.analyzer.analyze_state(conversation_history)
           
           # Predict optimal response strategy
           strategy = self.analyzer.recommend_strategy(current_state)
           
           return strategy

A/B Testing for Conversation Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Test different conversation strategies
   strategies = ["empathetic", "direct", "inquisitive"]
   
   results = {}
   for strategy in strategies:
       dataset = load_conversations(strategy=strategy)
       analyzer = TrajectoryAnalyzer(dataset)
       results[strategy] = analyzer.evaluate_strategy()
   
   # Statistical comparison
   best_strategy = compare_strategies(results)

This trajectory-based approach enables sophisticated analysis of conversational AI systems, providing insights into both micro-level turn quality and macro-level conversation outcomes. 