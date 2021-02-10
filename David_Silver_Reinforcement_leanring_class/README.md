# David Silver Reinforcement leanring class



## Abstract

note about Reinforcement leanring class lectured by David Silver



## Watch the website

See [bilibili](https://www.bilibili.com/video/BV1kb411i7KG).



## lecture 2: Markov Decision Processes

See courseware [Markov_Decision_Processes.pdf](./courseware_pdf/Markov_Decision_Processes.pdf).

实际上整个强化学习的过程就是在求解MDP（Markov Decision Processes）。

MDP是由Markov Processes到Markov Reward Processes，再到Markov Decision Processes，由此发展而来。

**MP**

MP的未来状态只依赖于当前状态，和过去无关（也可能是当前状态包含了所有过去状态）。

每个状态转移到下每一个继承态的概率是固定的，由State transition matrix P（见MDP.pdf第9页）来定义。

MP有State和State transition matrix组成。

MP是有限的过程而非无限的。

**MRP**

MRP在MP基础上新增了 reward function R和discount factor γ。注意MDP.pdf第10页中，花体字R代表reward function，普通字体R代表每一个继承态的奖励reward，reward function是算上State transition matrix P的平均reward（期望）（MDP.pdf在接下来的内容里几乎没有再提reward function，接下来出现的R基本代表的就是普通reward）。

The return Gt代表从时间步t开始，随机走完一个MP过程（同MDP.pdf中的说法随机采取的一个样本同意），获得的全部reward与discount factor相乘的和。

Value函数v（s）代表从时间步t开始算上State transition matrix P的平均Gt（期望）。

**Bellman Equation**

Bellman Equation就是把v（St）拆分成直接（立即）回报和t+1步的v（St+1）。

**MDP**

MDP在MRP基础上增加了finite set of actions A。

注意从这里开始，MDP.pdf中的学生过程的例子发生了变化，每个状态之间不再是按概率转移了，只有pub这个状态是按概率转移，当然现实世界的情况是，你的状态是由action和概率（环境）共同决定。

**Policy**

policy定义代理的行为；只由现在的状态决定；和时间步t无关。

**Value Function**

v（s）扩展为VΠ（s），Π（pai）即policy，现在v被认为是state-value function，由当前状态和policy共同决定。

新增action-value function QΠ（s，a），q相比v还由action决定。

依然可以用Bellman Equation拆分v和q，见（MDP.pdf30页-34页）。

**Optimal Value Function**

我们的终极目标就是求Optimal Value Function（就是求最优q），见MDP.pdf42页有每个state的最优q，可以思考是如何求出来的。

**Solving the Bellman Optimality Equation**

见MDP.pdf48页。

**MDP Extensions**

见MDP.pdf49-最后一页。没有看，随后补充。



## lecture 3: Planning by Dynamic Programming

See courseware [Planning_by_Dynamic_Programming.pdf](./courseware_pdf/Planning_by_Dynamic_Programming.pdf).

动态规划通常用于寻找问题的最优解，需要问题具备以下特点：

\- 寻找最优解的过程可以被分解为若干子问题

\- 子问题会被迭代若干次（意思是子问题可以以迭代的方式解决）

\- 子问题的解可以被储存并再使用（不用重复计算）

\- 最优解由子问题的最优解组合而成

\- DP更新每个state的value不是一步到位的，相当于要初始化每个state value，然后每次迭代更新state value是用上一次迭代的该state的后继state的期望值函数和立即回报（只在第一次迭代才用）进行更新

**Iterative Policy Evaluation**

\- 每次迭代都更新

\- 通过上一个state的state value function更新下一个继承态的状态值函数

在dp.pdf第10页的小方格例子中，迭代评价该policy即从每个state开始（左上右下state因环境动力学的原因被强制留在原地），因policy往上下左右的几率一样，因此遍历所有policy从某状态开始能达到的新状态，并求得分的平均（期望），即该网格的state value。

所以以Iterative Policy Evaluation要如何改进policy呢？即跟随state value最高的网格前进即可。

**Policy Iteration**

见dp.pdf第13页-19页，没有理解。

**Value Iteration**

这个过程是和Iterative Policy Evaluation刚好是反的，Iterative Policy Evaluation假设每一个state是起点，但Value Iteration假设每一个state是终点，反过来迭代。David Silver提到Intermediate value functions may not correspond to any policy（见dp.pdf第23页），因此该过程应该是等值迭代完了，找到起点，一步到位形成policy。

**以上方法总结**见dp.pdf第26页。从三种方法使用的Bellman Equation上即可看出区别。

**以上方法都属于同步动态规划（同步回调synchronous backups）**，同步很消耗计算资源，还有三种异步动规（Asynchronous Dynamic Programming）。

**Asynchronous Dynamic Programming**

**in-place dp**

见dp.pdf第29页，没有理解。

**Prioritised sweeping**

普通DP更新state的value function的顺序是随机的，要更新所有state的代价太大，因此出现了该方法，其为state的更新顺序排了序，以max（v（St+1）-v（St））到min（v（St+1）-v（St））（这里使用的似乎是v*（最大值，而非期望）称之为 **remaining Bellman error**）的顺序更新，因为按我们的直觉，state value变化大的更有可能是最佳路径。我们可以通过维护一个优先队列（ priority queue）的方式实现该方法。

**Real-Time Dynamic Programming**

该方法就是利用的引导policy action的value function不再是期望，而是policy真正走过的路径（maybe大部分情况下是最大值）。

**传统DP存在的问题**

DP uses full-width backups

\- 相当于每一次迭代要遍历所有继承态，因此相当于考虑到了所有sample，这样的空间复杂度太大，之后会使用抽样（比如model-free）的方式来进行，若使用抽样的方法，相当于我们就不知道环境动力学模型了（或环境模型，所以叫model-free）

\- 值函数也使用“逼近的值函数”

**contraction mapping theorem**

用来解释为什么我们这样做可以使值函数收敛到最优，并得到最优策略，见dp.pdf第29页-最后。未看。



## lecture 4: Model-Free Prediction

See courseware [Model_Free_Prediction.pdf](./courseware_pdf/Model_Free_Prediction.pdf).

model free的方法就是先采样，再从采样中进行学习。其实只有一类方法，但是有三种表现形式：

\- **MC**，Monte-Carlo Reinforcement Learning

\- **TD**，TD(0)，Temporal-Difference Learning

\- **TD(λ)**，ps: λ读作lambda

**MC**

从完整的episode中学习经验，其认为每个状态的optimal value等于E(G(s))，所以显而易见，采样次数越多，采样空间覆盖整个样本空间越广，Q(s)或V(s)越准确，不具有自举(bootstrapping)的功能。公式见mc-td.pdf13页。

**TD**

从完整或不完整的episode中学习经验，具有自举(bootstrapping)的功能，公式见mc-td.pdf16页。

**MC vs TD**

见mc-td.pdf21页。

简单来说：

\- MC是无偏估计，但方差大，TD方差小于MC，但是有偏，不过mean伴随迭代也在趋近真实的mean

\- MC的目标不是在优化Bellman Optimality Equation，只是在最小化样本的MSE，或者可以说只是在严格地拟合样本，所以实质上MC不算严格的MDP，而TD则是在优化Bellman Optimality Equation。所以当问题不在属于MDP时，MC的性能损失会较小

\- 另外理论上认为因为MC的目标为G(S(t)) = R(t + 1) + γR(t + 2) + ... + (γ ** (T - t - 1))R(T)，而TD为R(t + 1) + V(S(t + 2))，因为每个状态转移的直接回报与最优回报都有bias，所以MC累积了大量的bias

**Batch MC and TD**

当学习样本是有限时，我们应该反复利用这些样本去做GPI，类似训练神界网络的过程，即for epoch in range(len(any_number))内嵌套for episode in episodes。

**TD(λ)**

**Define the n-step return**

G(t)n = R(t + 1) + γR(t + 2) + ... + (γ ** (n - 1))R(t + n) + (γ ** n)V(S(t + n))

简单来说就是用Geometric weight对所有后续的G(t)n、G(t)(n + 1)...进行加权平均，见mc-td.pdf39页。

**Forward-view TD(λ)**

就是上面那种方法。

\- 实质上TD(**λ = 1**)就是MC

**Backward View TD(λ)**

使用**资格迹(Eligibility Traces)**的TD(λ)，见mc-td.pdf52页。

**TD(λ)的Offline updates和Online updates**

**没懂****，见**mc-td.pdf53-55页。



## lecture 5: Model-Free Control

See courseware [Model_Free_Control.pdf](./courseware_pdf/Model_Free_Control.pdf).

为了方便，控制问题基本用Q而不用V。

**On-policy**

基于GPI，需要注意策略必须既保持持续探索，又保证在逐步收敛至Π*(最优策略)（被称为Greedy in the Limit with Infinite Exploration (GLIE)，见control.pdf17页），为了保证这个条件可以使用ε-软性策略(ε-greedy属于一种，ps: ε读作epsilon)，其中ε按时间步逐渐变小，见control.pdf13页。

\- Sarsa，一种常用的同轨TD，还有Forward View Sarsa(λ)和Backward View Sarsa(λ)以及表格版的Sarsa(λ)实现，见control.pdf23-32页

**Off-policy**

通过评估一个行动策略来优化目标策略(均指Q)，因为两个策略的分布不同，所以要用一个策略的Q估算另一个Q需要使用重要度采样，见control.pdf36-38页。

**Q-Learning**

一种不需要重要度采样的方法，没懂，似乎通过改进目标策略，再把改进后的目标策略改为软性策略，进行采样，改进策略的方式是直接选取能最大化Q的action，见control.pdf39-42页。

**Relationship Between DP and TD**

DP和TD的回溯图和优化Q函数的关系，见control.pdf46-47页。



## lecture 6: Value Function Approximation

See courseware [Value_Function_Approximation.pdf](./courseware_pdf/Value_Function_Approximation.pdf).

思想是使用状态feature去代表状态，而非使用表格法（表格法属于特殊的函数逼近法，见fa.pdf第15页）去储存每个状态并为每个状态单独赋予一个状态编号，随后将状态输入模型，输出state value或者输出每个action value，或者将动作也特征化输入模型，直接输出该状态该动作的action value，详见fa.pdf第7页。

所以正是因为这种弊端，如果动作空间是连续的或者大型离散的，即无法用函数逼近 + epsilon greedy policy的方法。

MC因为是无偏估计，所以在on-policy的情况下一定是收敛的，但是除线性TD（可收敛到TD不动点，全局最优）外，非线性TD法不保证收敛（有发散的危险），强化学习导论里称其为致命三要素（见书260页）。

David Silver称Forward view and backward view linear TD(λ) are equivalent，见fa.pdf第20页。

**Experience Replay**

所谓经验重现就是一个batch的经验反复进行梯度下降（因为都没有验证集，甚至不知道该下降几个时间步）

对于线性模型的经验重现，最终会Converges to least squares solution

**DQN**

David Silver称DQN是十分稳定的，正是因为使用经验重现和固定target model的策略才如此稳定，且target model的param一般要几千个时间步才和policy model同步一次。所以torch 官方的DQN不代表是真实的DQN，应该去看看真实DQN的样子



## lecture 7: Policy Gradient

See courseware [Policy_Gradient.pdf](./courseware_pdf/Policy_Gradient.pdf).

批量梯度下降的名字一般是The batch gradient decent，还可以被称为The vanilla gradient decent

**同值函数方法的区别及优势**

见pg.pdf第4-5页

**Policy Objective Functions**

见pg.pdf第10页，David Silver介绍了3中策略目标函数，强化学习导论里仅有两种。

分别是

 \- start value used in episodic environments

 \- average value used in continuing environments

 \- average reward per time-step（书上没这种）

**Policy Optimisation（优化方法）**

见pg.pdf第11页，就我们熟悉了梯度法，近似梯度法（finite differences 有限差分），不用梯度的方法（如遗传算法）

**Policy Gradient Theorem（策略梯度定理）**

见pg.pdf第20页，这里和书上有不同，书上说策略梯度正比于那个公式，而David Silver直接写的等于，虽然也没有多大影响...

**Score Function**

见pg.pdf第16页，可以用Likelihood ratios转为对数策略梯度，最终形式就是为对数策略梯度， 常用的对数策略梯度有

 \- Softmax Policy用于离散动作

 \- Gaussian Policy用于连续动作

**策略梯度方法**

总览见pg.pdf最后一页

**Monte-Carlo Policy Gradient (REINFORCE)**

见pg.pdf第21-页，无偏，高方差，和MC值函数法一样收敛慢，减小方差的方法是加入baseline，见pg.pdf第29页

**Q Actor-Critic**

见pg.pdf第23-页，有偏，方差小于MC-PG，因为是有偏，所以may not find the right solution，见pg.pdf第26页，但是只要满足Compatible Function Approximation Theorem就能正确收敛，见pg.pdf第27页，

**Advantage Actor-Critic**

见pg.pdf第29-页，加入baseline后Score Function —— Advantage Function

**TD Actor-Critic**

见pg.pdf第31页，无偏差的，书上也是直接介绍的这种方法而没有介绍Q Actor-Critic，是目前最常用且收敛概率较大的方法，简单来说就是把action value func替换为state value func，且这种方法相当于自带了baseline

**TD(λ) Actor-Critic**

见pg.pdf第32-34页，有偏的

**Natural Actor-Critic**

见pg.pdf第35-页，David Silver提出的算法，既可以ascent，也可以decent，说的可以减小样本中地噪音对模型对学习policy gradient的危害。可以搜原文看看！



## lecture 8: Integrating Learning and Planning

See courseware [Integrating_Learning_and_Planning.pdf](./courseware_pdf/Integrating_Learning_and_Planning.pdf).

这一节课的内容和强化学习导论第八章内容相近

**Model-Based Reinforcement Learning**

其实就是先用真实的sample拟合出一个environment model，这个environment model(transection model)在输入state和action的时候会返回state和reward。这个env model通常可以由两个模型组成，一个可以是回归模型，根据输入state和action拟合输出reward，另一个是去拟合状态转移(state transitions)概率，可以用密度估计。

Learning s, a → r is a regression problem

Learning s, a → s' is a density estimation problem

**当env model不准确的时候**

可能必须考虑放弃model-base的方法或者了解env model不准确的原因

**Integrated Architectures**

**Model-Free RL**

\- No model

\- Learn value function (and/or policy) from real experience

**Model-Based RL (using Sample-Based Planning)**

\- Learn a model from real experience

\- Plan value function (and/or policy) from simulated experience

**Dyna**

\- Learn a model from real experience

\- Learn and plan value function (and/or policy) from real and simulated experience

**Dyna-Q Algorithm**

算法逻辑见dyna.pdf第27页

**Simulation-Based Search(Forward Search)**

**Simple Monte-Carlo Search**

见dyna.pdf第35页，使用一种预演策略(simulation policy)从**current(real) state**开始模拟至episode结束或当discount极小时停止，取所有的模拟episode平均回报作为当前状态动作二元组(a | current state)的动作价值

**Monte-Carlo Tree Search**

见dyna.pdf第36-37页，用文字很难解释其算法原理，dyna.pdf第43-46页可以帮助理解原理，根节点一定会模拟每个状态动作二元组(a | current state)的动作价值，并选择最大的那个，树叶子节点的q value一定是由**Default policy**(预演策略)(simulation policy)得到的，大部分MCTS都会保存叶子节点的q value

**Each simulation consists of two phases**

*- Tree policy*：沿着q value最大的叶子节点生长

*- Default policy*(预演策略)(simulation policy)：按既定策略模拟，就算是random uniform policy，整体MCTS的效果也会惊人

最终取所有叶子节点的q value平均值作为根状态动作二元组(a | current state)的动作价值(和树回溯更新一样)

**Dyna-2**

见dyna.pdf第52页



## lecture 9: Exploration and Exploitation

See courseware [Exploration_and_Exploitation.pdf](./courseware_pdf/Exploration_and_Exploitation.pdf).

这一节课的内容和强化学习导论第二章内容相近，大部分内容聚焦于多臂赌博机(multi-arm bandits)问题，multi-arm bandits can be regard as one-step(no state) decision-making problems(not reforcement learning, not MDP)(见xx.pdf第6页)，再稍微复杂一点multi-arm bandits变为contextual bandits(one-step decision-making problems with state)(提高CTR问题可以视为contextual bandits问题)，再复杂即MDP(reforcement learning)问题

开发和探索的本质

\- **Exploitation** Make the best decision given current information

\- **Exploration** Gather more information

主要分为**有以下方法大类**

\- Naive Exploration

\- Optimistic Initialisation

\- Optimism in the Face of Uncertainty

\- Probability Matching(也属于Optimism in the Face of Uncertainty)

\- Information State Search

**multi-arm bandits**

以下均是处于多臂赌博机场景

**Regret**

见xx.pdf第7-8页，定义了多臂赌博机需要优化的目标

greedy policy and epsilon greedy policy都只能得到linear regret, 见xx.pdf第9-11页，我们想要找到sublinear regret

**Optimistic Initialisation**

给每个动作的Q(a)初始化一个尽可能高的值，Encourages systematic exploration early on，但是最后的total regret还是linear的，见xx.pdf第12页

**Decaying ε****t****-Greedy Algorithm**

见xx.pdf第13页，若按这个规律动态调整**ε**即可得到sublinear regret，但需要知道gaps(最优q和现在q的差)

**Lower Bound**

见xx.pdf第14页，Lai and Robbins证明了算法的下线(最优)由gap和KL divergence定义

**Optimism in the Face of Uncertainty**

这类算法简而言之就是认为大的回报会出现在不确定性较高的动作上(N(t | a = a')较小)(Uncertain actions have higher probability of being max)

*Upper Confidence Bounds*

定义见xx.pdf第17页，这类算法的逻辑就是只在进行动作选择(Ignores uncertainty from policy improvement)时，在q value的基础上加上一个Ut(a = a')(衡量置信度的值)，*可以根据不同的Inequality来定义U**t**(a = a')，如：*

\- Hoeffding's Inequality

\- Bernstein's Inequality

\- Empirical Bernstein's Inequality

\- Chernoff Inequality

\- Azuma's Inequality

Hoeffding's Inequality

以Hoeffding's Inequality为例，见xx.pdf第18-20页，可以得到UCB1算法

Bayesian Bandits

不由Inequality来定义Ut，而是假设先验分布并求后验分布的方式，见xx.pdf第22-23页，Bayesian UCB Example: Independent Gaussians

*Probability Matching*

见xx.pdf第24-25页，**Thompson sampling** implements probability matching, achieves Lai and Robbins lower bound! 不太懂

**Information State Search**

总的来说运用了信息增益的思想，持续尝试不确定性较高的动作(N(t | a = a')较小)的信息增益较大(Information gain is higher in uncertain situations)(见xx.pdf第26页)，augmented information state space本身是一个MDP(见xx.pdf第27页)

*Bernoulli Bandits*

见xx.pdf第28-32页

**Contextual Bandits**

**Linear Regression in Contextual Bandits**

见xx.pdf第34页，用最小二乘来解该线性方程

*Linear Upper Confidence Bounds*

见xx.pdf第35-37页，简而言之就是因为估计本身存在误差，因此最小二乘得出来的参数并非最优参数，但最优参数大概率落在以计算参数为中心的一个置信区间内，通过这种方法我们可以计算出Uθ(s, a)(just like Ut(a = a') in multi-arm bandits)

**Exploration/Exploitation Principles to MDPs**

*Optimistic Initialisation in MDP*

见xx.pdf第40-41页，model-based不太懂

*Upper Confidence Bounds in MDP*

见xx.pdf第42页，不太懂

*Bayesian Model-Based RL*

见xx.pdf第43-44页，Thompson Sampling不太懂

*Information State Search in MDPs*

不太懂