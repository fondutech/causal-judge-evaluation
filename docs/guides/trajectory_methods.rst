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

Working with Trajectories
-------------------------

CJE supports trajectory evaluation through the trajectory dataset and estimators:

.. code-block:: python

   from cje.data import TrajectoryJSONLDataset
   from cje.estimators import get_estimator
   
   # Load trajectory data
   dataset = TrajectoryJSONLDataset("trajectories.jsonl")
   
   # Use trajectory-aware estimator
   estimator = get_estimator("DRCPO", trajectory_mode=True)
   estimator.fit(dataset)
   results = estimator.estimate()

**Note**: The trajectory support in CJE is primarily designed for reinforcement learning-style agent trajectories with states and actions, not conversational trajectories. For multi-turn conversation analysis, structure your data as individual examples with full conversation context.

Policy Comparison
~~~~~~~~~~~~~~~~~

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

Advanced Configuration
----------------------

Trajectory-Specific Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configure trajectory evaluation in your YAML:

.. code-block:: yaml

   # Trajectory configuration
   trajectory:
     max_turns: 10                 # Limit conversation length
     context_window: 3             # Include last 3 turns as context
     step_aggregation: "mean"      # How to combine step rewards

**Note**: These trajectory features are designed for RL-style agent trajectories. For conversational analysis, consider structuring your data as individual examples with conversation history included in the context.

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