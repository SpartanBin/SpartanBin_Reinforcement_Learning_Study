# Surrogate objective optimization



## Abstract

Note about this algorithm is to optimize the surrogate objective(surrogate function) instead of target function.



## Method iteration trajectories

### trajectory 1 note

- [CPI 2002] [Approximately optimal approximate reinforcement learning.pdf](./paper/Approximately optimal approximate reinforcement learning.pdf)
- [TRPO 2015] [Trust Region Policy Optimization.pdf](./paper/Trust Region Policy Optimization.pdf)
- [PPO 2017] [Proximal Policy Optimization Algorithms.pdf](./paper/Proximal Policy Optimization Algorithms.pdf)

#### notation about importance sampling:

- Can see the [lesson teaching by Li Hongyi](https://www.bilibili.com/video/BV1MW411w79n?p=2) for more details
- We want to calculate expected **f(x)**, where **x** sampled from probability distribution **p**, **Ex~p[f(x)]**. But we only have **x** sampled from probability distribution **q**. You can use importance sampling to calculate **Ex~p[f(x)] = ∫x~q[f(x) * p(x) / q(x) * q(x)]dx ≈ Ex~q[f(x) * p(x) / q(x)]**
- **Issue of importance sampling**: using importance sampling to calculate expected **f(x)** is unbiased, but the variance of **f(x)**, where **x** sampled from probability distribution **p** is different from the variance of **f(x)**, where **x** sampled from probability distribution **q**, even very different. So when your number of sampling is not enough, **Ex~p[f(x)]** may distinct obviously from **Ex~q[f(x)]**
- So if we do not add constraints or penalty for parameters updating in off-policy reinforcement learning, especially when **p**(you can see this as action model's policy) is vary different from **q**(you can see this as target model's policy), it may be resulting in Catastrophically worse due to wrong huge parameters updating. It generates TRPO and PPO from this problem

#### notation: 

- Let **η(π)** denote expected discounted reward, **η(π) = Eπ[Σt(γt * r(St))]**
- **Qπ** denote  state action value function, the value function **Vπ** and  the advantage function **Aπ**, **Aπ(s, a) = Qπ(s, a) -  Vπ(s)**

#### [CPI 2002] proves that: 

- **η(π') = η(π) + Eπ'[Σt(γt * Aπ(st, at))]**

#### notation: 

- Let **ρπ(s)** denote discounted visitation frequencies, **ρπ(s) = P(s0 = s) + γ * P(s1 = s) + γ ^ 2 * P(s2 = s)  + ... + γ ^ n * P(sn = s)**
- So **η(π')** can be rewrited as **η(π') = η(π) + Σs[ρπ'(s) * Σa(π'(a | s) * Aπ(s, a))]**
- So a nonnegative expected advantage per for any state, like **Σa(π'(a | s) * Aπ(s, a)) >= 0**, can guarantee nondecreasing(sometimes increasing) policy in policy iteration. Due to estimation error in function approximation, cannot guarantee all nonnegative expected advantage for any state

- Then we ignore discounted visitation frequencies changing in state visitation, **η(π') (changing notation to Lπ(π'))** transforming to **Lπ(π') = η(π) + Σs[ρπ(s) * Σa(π'(a | s) * Aπ(s, a))]**

#### [CPI 2002] proves that: 

- When parameterized **πθ** `in first order`, **Lπθ0(πθ0) = η(πθ0)**

#### [CPI 2002] proposes the method "conservative policy iteration": 

- Let **π' = argmaxπ'Lπold(π')**. The new policy **πnew** was defined to be, **πnew(a|s) = (1 - α) * πold(a|s) + α * π'(a|s)**, where **α** is a constant
- The method derived the following lower bound, **η(πnew) ≥ Lπold(πnew) - (2 * γ * ε) / ((1 - γ) ^ 2) * α ^ 2**, where **ε = maxs|Ea~π'[Aπ(s, a)]|**, and **α** is a constant
- [TRPO 2015] said "conservative policy iteration" is unwieldy and restrictive in practice, and it is desirable for a practical policy update scheme to be applicable to all general stochastic policy classes

#### notation: 

- by replacing **α** with a distance measure between **π** and **π'**, and changing the constant **ε** appropriately and using  total variation divergence, defined by **DTV(p || q) = 1 / 2 * Σi(|pi - qi|)** for discrete probability distributions **p**, **q**
- We defines total variation divergence **DTV(p || q)**, **DTV(p || q) = 1 / 2 * Σi(|pi - qi|)**, where **p**, **q** is discrete probability distributions
- And defines **DmaxTV(π, π')** as **DmaxTV(π, π') = maxs(DTV(π(any_a | s) ||  π'(any_a | s)))**

#### [TRPO 2015] proves that:

- Let α = DmaxTV(πold, πnew), then the following bound holds: **η(πnew) ≥ Lπold(πnew) - (4 * γ * ε) / ((1 - γ) ^ 2) * α ^ 2**, where **ε = maxs|Aπ(s, a)|**
- Because of **DTV(p || q) ^ 2 ≤ DKL(p || q)**[Pollard 2000], bound transforms to **η(π') ≥ Lπ(π') - C * DmaxKL(π, π')**, where **C = (4 * γ * ε) / ((1 - γ) ^ 2)**
- We assume exact evaluation of the advantage values **Aπ**. Thus, by maximizing Mi at each iteration, we guarantee
  that the true objective **η** is non-decreasing. **Lπ(π') - C * DmaxKL(π, π')** is called `surrogate function`

#### [TRPO 2015] proposes the method "Trust Region Policy Optimization": 

- [TRPO 2015] said that if we used the penalty coefficient C recommended by the theory above, the step sizes would be very
  small.(it means optimizing `surrogate function` directly)
- [TRPO 2015] said other impractical method is "maximize **Lθold(θ)**, subject to **DmaxKL(θold, θ) <= δ**", because of the large number of constraints
- Then they proposes method witch uses `average KL divergence`, defined as **D'ρKL(θ1, θ2) = Es~ρ[DKL(πθ1(any_a | s) ||  πθ2(any_a | s))]**, "maximize **Lθold(θ)**, subject to **D'ρθoldKL(θold, θ) <= δ**". They said it's performance like the impractical method above
- Practical Algorithm repeatedly performs the following steps: `A`. collect a set of state-action pairs along with any estimates of their *Q*-values; `B`. by averaging over samples, construct the estimated objective and constraint blow, "maximize **Es~ρθold, a~q[πθ(a | s) / q(a | s) * Qθold(s, a)]**, subject to **Es~ρθold[DKL(πθold(any_a | s) ||  πθ(any_a | s))]**"; `C`. approximately solve this constrained optimization problem to update the policy’s parameter vector **θ**. In the article, they use `conjugate gradient`, reading for more detail

#### [PPO 2017] algorithm 1 Adaptive KL Penalty Coefficient: 

- [PPO 2017] said the theory justifying TRPO actually suggests using a penalty instead of a constraint, like solving the unconstrained optimization problem, "maximize **Et[(πθ(at | st) / πθold(at | st)) * At - β * KL[πθold(· | st), πθ(· | st)]]**", instead of constraint. But it is hard to choose a single value of **β** that performs well across different problems—or even within a single problem, where the the characteristics change over the course of learning
- Algorithm has two parts: `A`. using several epochs of minibatch optimizer, optimize the KL-penalized objective, **Et[(πθ(at | st) / πθold(at | st)) * At - β * KL[πθold(· | st), πθ(· | st)]]**; `B`. compute **d = Et[KL[πθold(· | st), πθ(· | st)]]**, If **d < dtarg / 1.5, β ← β/2**, or If **d > dtarg * 1.5, β ← β × 2**
- `Question`: what is **dtarg**?(author said we achieve some target value of the KL divergence **dtarg** each policy update)

#### [PPO 2017] algorithm 2 Clipped Surrogate Objective: 

- Author said algorithm 2 performs better than algorithm 1
- Let **rt(θ)** denote the probability ratio, **rt(θ) = πθ(at | st) / πθold(at | st)**. Surrogate objective without constraint is **Et[rt(θ) * At]**. Without a constraint, maximization of surrogate objective would lead to an excessively large policy
  update. So we consider how to modify the objective, to penalize changes to the policy that move **rt(θ)** away from 1.
- We transform surrogate objective function to **Lclip(θ) = Et[min(rt(θ), clip(rt(θ), 1 - ε, 1 + ε)) * At]**, **ε** being a hyperparameter, empirically equivalent to 0.1 or 0.2