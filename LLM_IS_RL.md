# Token to Intelligence: Understanding LLM Capabilities Through Average Reward

## Abstract
Large Language Models (LLMs) present an intriguing paradox: while they are trained solely to predict the next token in a sequence, they somehow develop sophisticated capabilities like reasoning, problem-solving, and maintaining long-term coherence. How does such a myopic training objective lead to such broad intelligence? This paper resolves this puzzle by proving that LLM training is mathematically equivalent to average reward optimization in reinforcement learning when learning from human-generated text. Much like how predicting each move in a grandmaster's chess game implicitly teaches overall strategic principles, we show that correctly predicting each token in human-written text implicitly optimizes for human-level language capabilities. This equivalence explains why models trained on local token prediction naturally develop global coherence and sophisticated behaviors: they are implicitly optimizing long-term performance under a stationary distribution of optimal (human) behavior. Our framework provides theoretical foundations for why LLMs work, with implications for model training, data curation, and architectural design.

## 1. Introduction

Language models are trained on a seemingly simple task: predict the next token in a sequence based on previous tokens. This training objective appears remarkably limited - how can predicting the next word lead to complex capabilities like reasoning, problem-solving, and maintaining coherent conversations over long contexts? This apparent disconnect between the simplicity of the training objective and the sophistication of the resulting behavior has lacked rigorous theoretical justification.
We propose a novel perspective that resolves this paradox: LLM training is mathematically equivalent to optimizing the average reward in reinforcement learning when training on human-generated text. Consider an analogy: imagine learning to play chess by watching grandmasters play. You could either:

- Try to predict each individual move they make (analogous to token prediction)
- Try to understand their overall strategy to win the game (analogous to reinforcement learning)

Our key contribution is proving that, under certain conditions, these two approaches are mathematically equivalent. Just as correctly predicting each move in enough grandmaster games would implicitly teach you winning chess strategies, we show that correctly predicting each token in human-written text implicitly optimizes for human-level language capabilities.
This perspective offers several crucial insights:

- It explains how local token prediction can lead to global coherence and sophisticated behavior
- It provides theoretical justification for why training on high-quality human text leads to human-like capabilities
- It suggests new ways to think about model training and optimization

## 2. Background

### 2.1 Average Reward in Reinforcement Learning
In reinforcement learning, an agent learns to make decisions by interacting with an environment. While many RL formulations focus on maximizing discounted future rewards, there's an alternative perspective: optimizing the average reward received over time.

The key insight of average reward optimization is that it converts a long-term optimization problem into a short-term one. Rather than explicitly calculating future rewards, it shows that under a stationary distribution, optimizing immediate rewards naturally leads to optimal long-term behavior. This perspective is particularly powerful for analyzing language models. While LLMs appear to only optimize for immediate next-token prediction, they somehow achieve sophisticated long-term capabilities. As we will show, this apparent paradox can be resolved by viewing LLM training as average reward optimization, where the stationary distribution of human text allows local token optimization to yield global language capabilities.

Formally, in the average reward framework, we consider a policy π operating in a stationary environment. At each step:

The agent observes a state s from a stationary distribution μ(s)
It takes an action a according to its policy π(a|s,θ)
This leads to a new state s' and an immediate reward r
This process continues indefinitely

The mathematical objective is to optimize the expected reward per step:

$$
r(\pi) = \nabla \sum_s \mu(s) \sum_a \pi(a | s, \theta) \sum_{s', r} p(s', r | s, a) r
$$

where:

$\mu(s)$ represents the stationary distribution over states (how often each state is visited)
$\pi(a | s, \theta)$ is the policy function (how the agent chooses actions)
$p(s', r | s, a)$ captures the environment dynamics (what happens after each action)
$r$ is the immediate reward

Unlike discounted reward formulations that require complex estimations of future returns (Q-functions), the average reward framework elegantly focuses on immediate rewards under a stable state distribution. This simplicity will be key to establishing its connection to language model training.

### 2.2 Language Model Training

In LLM training, we maximize:

$$
\mathcal{L}(\theta) = \mathbb{E}\_{x \sim \mathcal{D}} [\log P(x_t|x_{\lt t}, \theta)]
$$

The gradient of $\mathcal{L}(\theta)$ with respect to $\theta$ is:

$$
\nabla_{\theta} \mathcal{L}(\theta) = \mathbb{E}\_{x \sim \mathcal{D}} [\nabla_{\theta} \log P(x_t|x_{\lt t}, \theta)]
$$

where $\mathcal{D}$ represents the distribution of human-generated text.

## 3. The Equivalence Theorem

### 3.1 Key Conditions

Two critical conditions enable the equivalence:

1. **Stationarity**:
    - When sampling from the fixed training dataset $\mathcal{D}$, we obtain a stationary distribution over states (contexts)
    - The induced state distribution $\mu(s)$ remains constant because:
      - Our dataset $\mathcal{D}$ is fixed
      - We sample randomly and uniformly from $D$
      - Each training example is sampled independently
    - This matches the RL setting where following a fixed policy $\pi$ induces a stationary state distribution $\mu_\pi(s)$
2. **Optimal Policy Samples**:
    - Human-generated text represents samples from a near-optimal policy
    - This assumption is supported by humans' demonstrated mastery of language. We acknowledge this is an idealization and discuss its limitations in later sections.
3. **Markov Property in Context Windows**:
    - Given a context window of size $k$, future tokens depend only on this context
      - $P(x_t|x_{\lt t}) = P(x_t|x_{t-k:t-1})$
    - This is a practical limitation of current LLM architectures

### 3.2 Formal Equivalence

**Theorem**: Under the conditions above, maximizing the LLM training objective is equivalent to optimizing the average reward criterion.

*Proof*:

1) First, let's consider the average reward gradient:

$$
\nabla_\theta r(\pi) = \nabla_\theta \sum_s \mu(s) \sum_a \pi(a | s, \theta) \sum_{s', r} p(s', r | s, a) r
$$

2)  Establish the mapping between RL and LLM components:
   - States $s$ represents context $x_{<t}$
   - Actions $a$ represents next token $x_t$
   - State distribution $\mu(s)$ is $P_\mathcal{D}(x_{<t})$, the probability of seeing context $x_{<t}$ in the training data
   - Policy $\pi(a|s,\theta)$ is $P(x_t|x_{<t},\theta)$, the model's token prediction
   - $p(s',r|s,a)$ represents the transition dynamics where:
     - The next state $s'$ includes both the extended context $[x_{<t};x_t]$
     - The reward $r$ depends on $P_\text{human}(x_t|x_{<t})$
   - Reward $r$ is $\log P_\text{human}(x_t|x_{<t})$, the log probability under the optimal (human) policy

3) Substituting these in:

$$
\nabla_{\theta} r(\pi) = \nabla_{\theta} \sum_{x_{\lt t}} P\_\mathcal{D}(x_{\lt t}) \sum_{x_t} P(x_t|x_{\lt t},θ) \sum_{x_{t+1}, r} p(x_{t+1}, r|x_{\lt t}, x_t) r
$$

4) By the stationarity assumption, we can simplify the transition dynamics:

**Lemma 1**: Under the stationarity assumption, for any context $x_{\lt t}$ and token $x_t$:

$$
\sum_{x_{t+1}, r} p(x_{t+1}, r|x_{\lt t}, x_t) r = \log P_{human}(x_t|x_{\lt t})
$$

5) Applying Lemma 1:

$$
\nabla_{\theta} r(\pi) = \nabla_{\theta} \sum_{x_{\lt t}} P\_\mathcal{D}(x_{\lt t}) \sum_{x_t} P(x_t|x_{\lt t},\theta) \log P\_\text{human}(x_t|x_{\lt t})
$$

4) By the definition of expectation:

$$
\nabla_{\theta} r(\pi) = \nabla_{\theta} \mathbb{E}\_{x_{\lt t} \sim \mathcal{D}} \mathbb{E}\_{x_t \sim P\_\text{human}(\cdot|x_{\lt t})} [\log P(x_t|x_{\lt t},\theta)]
$$

5) Using the fact that we sample $(x_{<t}, x_t)$ pairs directly from the training data $\mathcal{D}$, this simplifies to:

$$
\nabla_{\theta} r(\pi) = \mathbb{E}\_{x \sim \mathcal{D}} [\nabla_{\theta} \log P(x_t|x_{\lt t}, \theta)]
$$

This final form is exactly the gradient of the LLM training objective.

**Proof of Lemma 1**:
1. When sampling from our fixed dataset $\mathcal{D}$, we have a stationary distribution over states (contexts) $\mu(x_{\lt t})$

2. For each context-token pair $(x_{\lt t}, x_t)$ in our dataset:
   * The reward $r$ is defined as $$r = \log P_{human}(x_t|x_{\lt t})$$
   * The next context $x_{t+1}$ is deterministically obtained by appending $x_t$ to $x_{\lt t}$
   * Therefore $p(x_{t+1}, r|x_{<t}, x_t) = p(x_{t+1}|x_{<t}, x_t)$ when $r = \log P_{human}(x_t|x_{<t})$ and 0 otherwise

3. Since $$\sum_{x_{t+1}} p(x_{t+1}|x_{\lt t}, x_t) = 1$$ (transition probabilities sum to 1)

4. The summation reduces to:

   $$\sum_{x_{t+1}, r} p(x_{t+1}, r|x_{\lt t}, x_t) r = \log P_{human}(x_t|x_{\lt t}) \sum_{x_{t+1}} p(x_{t+1}|x_{\lt t}, x_t) = \log P_{human}(x_t|x_{\lt t})$$

This shows that under our sampling process from the fixed dataset, the expected reward for a state-action pair is exactly the log probability under the human policy.

### 3.3 Implications

This equivalence explains several phenomena:

1. **Convergence to Human-Level Performance**: 
   Since we're optimizing average reward using samples from an optimal policy (human text), the model naturally converges toward human-level performance.

2. **Local to Global Coherence**: 
   Though we only use immediate rewards (next-token prediction), the stationarity of the distribution ensures global coherence.

3. **Role of Data Quality**:
   The quality of the training data directly affects performance because it determines the optimal policy we're learning from.

## 4. Discussion

The average reward perspective provides several insights:

1. **Training Data as Expert Demonstrations**:
   - Human-generated text serves as demonstrations of optimal behavior
   - The stationary distribution ensures we learn global patterns

2. **Importance of Data Quality**:
   - Training data quality directly affects the optimal policy we learn from
   - This explains why careful data curation improves model performance

3. **Beyond Token Prediction**:
   - While the immediate reward is based on next-token prediction
   - The stationary distribution ensures we capture long-range dependencies

## 5. Conclusion

By establishing the equivalence between LLM training and average reward optimization, we provide a theoretical framework for understanding how local token prediction leads to global coherence and human-level capabilities. This perspective offers new ways to think about improving LLM training and suggests directions for future research.
