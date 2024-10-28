# Reframing Language Model Training as Average Reward Optimization in Reinforcement Learning

## Abstract
Large Language Models (LLMs) have demonstrated remarkable human-level capabilities despite being trained on seemingly myopic next-token prediction. This apparent paradox - how local token prediction leads to global coherence and sophisticated behaviors - has lacked rigorous theoretical justification. In this paper, we resolve this puzzle by proving that LLM training is mathematically equivalent to average reward optimization in reinforcement learning when learning from human-generated text. This equivalence explains why models trained purely on next-token prediction naturally converge to human-level capabilities: they are implicitly optimizing long-term performance under a stationary distribution of optimal (human) behavior. Our framework provides theoretical foundations for why LLMs work, with implications for model training, data curation, and architectural design. This result bridges the gap between the simplicity of token-level training and the emergence of sophisticated language capabilities.

## 1. Introduction

Language models are typically trained to predict the next token in a sequence based on a large corpus of text data. While this is often viewed as a supervised learning problem, we propose an alternate perspective: that LLM training is mathematically equivalent to optimizing the average reward in reinforcement learning when training on human-generated text.

This perspective offers several insights:
1. It provides a theoretical framework for understanding why LLMs converge to human-level performance
2. It explains how local token prediction can lead to global coherence
3. It suggests new ways to think about LLM training and optimization

## 2. Background

### 2.1 Average Reward in Reinforcement Learning
In reinforcement learning, an agent learns to make decisions by interacting with an environment. While many RL formulations focus on maximizing discounted future rewards, there's an alternative perspective: optimizing the average reward received over time. This approach is particularly relevant when we care about long-term sustainable performance rather than rewards accumulated from a specific starting point.
To understand average reward, consider a simple example: imagine a robot learning to navigate a building. While discounted reward might encourage finding the quickest path to a specific goal, average reward optimization would focus on maintaining efficient movement patterns over extended periods. The robot would learn behaviors that work well consistently, much like how humans develop natural walking gaits.
In language, this parallels how humans maintain coherent communication over time, rather than optimizing for individual sentences in isolation. This connection will be crucial for our analysis of language models.
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

where $\mathcal{D}$ represents the distribution of human-generated text.

## 3. The Equivalence Theorem

### 3.1 Key Conditions

Two critical conditions enable the equivalence:

1. **Stationarity**: The training data distribution $\mathcal{D}$ represents a stationary distribution over contexts (states)
2. **Optimal Policy Samples**: The training data consists of samples from an optimal policy (human-generated text)

### 3.2 Formal Equivalence

**Theorem**: Under the conditions above, maximizing the LLM training objective is equivalent to optimizing the average reward criterion.

*Proof*:

Let's establish the mapping:
- States ($s$) correspond to token contexts ($x_{<t}$)
- Actions ($a$) correspond to next tokens ($x_t$)
- The stationary distribution $\mu(s)$ corresponds to the distribution of contexts in $\mathcal{D}$
- The policy $\pi(a|s,\theta)$ corresponds to the LLM's token prediction $P(x_t|x_{<t},\theta)$
- The immediate reward $r$ corresponds to the log probability under the optimal (human) policy

The average reward gradient is:

$$
\nabla_\theta r(\pi) = \nabla_\theta \sum_s \mu(s) \sum_a \pi(a | s, \theta) \sum_{s', r} p(s', r | s, a) r
$$

Under our mapping, let's derive this step by step:

1) First, let's consider the average reward gradient:

$$
\nabla_\theta r(\pi) = \nabla_\theta \sum_s \mu(s) \sum_a \pi(a | s, \theta) \sum_{s', r} p(s', r | s, a) r
$$

2) In the language modeling context:
   - $s$ represents context $x_{<t}$
   - $a$ represents next token $x_t$
   - $\mu(s)$ is $P_\mathcal{D}(x_{<t})$, the probability of seeing context $x_{<t}$ in the training data
   - $\pi(a|s,\theta)$ is $P(x_t|x_{<t},\theta)$, the model's prediction
   - $p(s',r|s,a)$ becomes deterministic: the next state $s'$ is always $[x_{<t};x_t]$
   - $r$ is $\log P_\text{human}(x_t|x_{<t})$, the log probability under the optimal (human) policy

3) Substituting these in:

$$
\nabla_{\theta} r(\pi) = \nabla_{\theta} \sum_{x_{\lt t}} P\_\mathcal{D}(x_{\lt t}) \sum_{x_t} P(x_t|x_{\lt t},\theta) \log P\_\text{human}(x_t|x_{\lt t})
$$

4) Since we're training on human-generated data, $P_\text{human}(x_t|x_{<t})$ is the empirical distribution in our training data. Therefore:

$$
\nabla_{\theta} r(\pi) = \nabla_{\theta} \mathbb{E}\_{x_{\lt t} \sim \mathcal{D}} \mathbb{E}\_{x_t \sim P\_\text{human}(\cdot|x_{\lt t})} [\log P(x_t|x_{\lt t},\theta)]
$$

5) Using the fact that we sample $(x_{<t}, x_t)$ pairs directly from the training data $\mathcal{D}$, this simplifies to:

$$
\nabla_{\theta} r(\pi) = \mathbb{E}\_{x \sim \mathcal{D}} [\nabla_{\theta} \log P(x_t|x_{\lt t}, \theta)]
$$

This final form is exactly the gradient of the LLM training objective. The key insight is that by training on human-generated text (optimal policy) and maintaining stationarity through the training data distribution, we are effectively optimizing the average reward criterion.

Which is exactly the gradient of the LLM training objective.

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
