---
layout: default
title: "Learning Undirected Graphical Models"
subtitle: Undirected graphical models formed a large part of the initial push for
          machine intelligence, and remain relevant today. Here, I motivate and 
          derive Monte Carlo-based learning algorithms for such models.
posted: 2020-09-06
updated: 2020-09-06
keywords: machine intelligence, generative models
published: true
---

## Introduction

Graphical models aim to define (and subsequently learn) a probability
distribution over a set of hidden and observed variables. These variables are
represented as nodes, with edges defining modeled interactions between the
variables. Directed graphical models (such as the Bayesian network) allow for
the principled definition of distributions over these variables by defining
directed relationships between nodes, while undirected models (such as the
Markov random field) define an *energy function* modeling undirected
interactions between nodes and compute a normalization function to define a
probability distribution from the energy function. 

Learning directed models is often performed with maximum likelihood estimation,
which has a [closed form
solution](https://ermongroup.github.io/cs228-notes/learning/directed/) in fully
observed Bayesian networks and requires more nuanced techniques including
[expectation--maximization and variational
learning](https://mananshah99.github.io/blog/2020/06/23/lvm/) for partially
observed networks involving latent variables.[^1] Both of these techniques
leverage that $p(x)$ is defined in terms of the conditional probability
distributions specified in the directed graphical model.

Learning undirected models, however, is more challenging as the lack of directed
connections makes it impossible to define conditional probability distributions
(and therefore directly define $p(x)$ in terms of nodes and edges). Instead,
undirected models define an energy function $\tilde{p}(x; \theta)$ over their
constituent nodes and additionally compute a normalization (also called
paritition) function $Z(\theta) = \sum_x \tilde{p}(x; \theta)$ so that
$\tilde{p}(x; \theta) / Z(\theta)$ defines a probability distribution. Since
the partition function require summing over all node values, it's often
intractable, making learning more challenging. 

In this post, we'll delve into working with the partition function, motivating
maximum likelihood-based learning algorithms for undirected graphical models.
In particular, we'll focus on the contrastive divergence algorithm, known for
its ability to efficiently train Restricted Boltzmann Machines (RBMs). The
material in this post is significantly inspired and derived from Section 18.1
in {% cite goodfellow2016deep %}; the reader is recommended to peruse the
chapter for related contents and additional details.

## Working with the Partition Function

Undirected graphical models define an unnormalized probability distribution
(also called an energy function) $\tilde{p}(x, z; \theta)$ over cliques of
variables in the graph, where $x$ collects the observed variables and $z$
collects hidden variables. In order to obtain a valid probability distribution,
we must normalize $\tilde{p}$, so that

$$
p(x, z; \theta) = \frac{1}{Z(\theta)} \tilde{p}(x, z; \theta)
$$

where

$$
Z(\theta) = \sum_{x, z} \tilde{p}(x, z; \theta)
$$

for discrete $x$ and $z$; the continuous analog simply requires replacing the
sum with an integral. It's immediately obvious that this operation is
intractable for most interesting models. While some models are designed with 
the express purpose of simplifying the partition function, we will not bother
ourselves with such specialized structures here; we'll instead focus on
training models with intractable $Z(\theta)$. 

### Decomposing the Likelihood

We'll sidestep the question of inference with an intractable partition function
and instead focus on the task of learning. The principle of maximum likelihood
tells us that we should maximize the probability of the observed data given
our model; a canonical way to do so is by gradient descent. In particular,
we have

$$
\begin{align*}
\nabla_\theta \log p(x; \theta) &= \nabla_\theta \log \sum_z p(x, z; \theta) \\
                                &= \nabla_\theta \log \sum_z \left( \frac{ \tilde{p}(x, z; \theta)}{Z(\theta)} \right) \\
                                &= \underbrace{\nabla_\theta \log \sum_z \tilde{p}(x, z; \theta)}_{\text{Positive phase}} \underbrace{- \nabla_\theta \log Z(\theta)}_{\text{Negative phase}}
\end{align*}
$$

This decomposition into a *positive* and *negative* phase of learning is well-known; we'll
have more to say about the interpretation of each phase as we continue our derivation. 
For now, let's look at each component in turn.

*Positive Learning Phase.* For our models of interest, we can reasonably assume
that summing over all values of $z$ is an intractable operation. However, we can
approximate the positive phase term with importance sampling and a variational
lower bound by following the techniques used for latent variable models 
described [here](https://mananshah99.github.io/blog/2020/06/23/lvm/). Specifically,
we begin by rewriting the postive phase as an expectation:

$$
\begin{align*}
\nabla_\theta \log \sum_z \tilde{p}(x, z; \theta) &= \nabla_\theta \sum_z q(z \mid x) \frac{\tilde{p}(x,z ; \theta)}{q(z \mid
x)} \\
&= \nabla_\theta \log \mathbf{E}_{z \sim q(z \mid x)} \left[ \frac{\tilde{p}(x,z ; \theta)}{q(z \mid
x)} \right]
\end{align*}
$$

To move the logarithm inside the expectation, we apply Jensen's inequality,
noting that equality is preserved if $q(z \mid x) = p(z \mid x; \theta)$ (where
we fix $\theta$ for the $q$ distribution).[^2] Fixing $q(z \mid x) = p(z \mid x;
\theta)$, we have that 

$$
\begin{align}
\nabla_\theta \log \sum_z \tilde{p}(x, z; \theta) &= \nabla_\theta \mathbf{E}_{z \sim q(z \mid x)} \log \tilde{p}(x,z ; \theta) - \nabla_\theta \mathbf{E}_{z \sim q(z \mid x)} \log  {q(z \mid x)} \\
&= \nabla_\theta \mathbf{E}_{z \sim p(z \mid x)} \log \tilde{p}(x, z; \theta)
\end{align}
$$

Which has the intuitive interpretation of maximizing the energy function of the
undirected model with respect to likely "completions" $z \sim p(z \mid x)$ of
the hidden variables given examples $x$ from the true data distribution. The
positive learning phase therefore increases the likelihood of true data in the
model. For directed graphical models (with no partition function), this is all
we need to do: the positive learning phase can be performed with
expectation-maximization in the case where $p(z \mid x)$ is tractable and can be
approximated with variational methods for intractable $p(z \mid x)$. For models
with a partition function, we need to understand the negative learning phase. 

*Negative Learning Phase.* The negative learning phase consists of computing
the gradient of the partition function with respect to parameters $\theta$. 
Looking more closely at this gradient (and omitting the $\theta$ argument of
$Z(\theta)$ for clarity), we have

$$
\begin{align*}
\nabla_\theta \log Z &= \frac{\nabla_\theta Z}{Z} \\
&= \frac{\nabla_\theta \sum_{x, z} \tilde{p}(x, z)}{Z} \\
&= \frac{\sum_{x, z} \nabla_\theta \tilde{p}(x, z)}{Z}
\end{align*}
$$

With the mild assumption that ${p}(x, z) > 0$ for all $x, z$, we can substitute
$\tilde{p}(x, z) = \exp(\log \tilde{p}(x, z))$ to obtain

$$
\begin{align}
\nabla_\theta \log Z &= \frac{\sum_{x, z} \nabla_\theta \exp(\log \tilde{p}(x, z))}{Z} \\
                    &= \frac{\sum_{x, z} \exp(\log \tilde{p}(x, z)) \nabla_\theta \exp(\log \tilde{p}(x, z))}{Z} \\
                    &= \sum_{x, z} \left( \frac{\tilde{p}(x, z)}{Z} \nabla_\theta \exp(\log \tilde{p}(x, z)) \right) \\
                    &= \sum_{x, z} p(x, z) \nabla_\theta \exp(\log \tilde{p}(x, z)) \\
                    &= \mathbf{E}_{x, z \sim p(x, z)} \nabla_\theta \log \tilde{p}(x, z)
\end{align}
$$

which allows us to express the negative learning phase as an expectation over
$x$ and $z$ drawn from the model distribution. 

### Putting it all Together

Our work in the prior sections allowed us to re-express the positive and negative
learning phases as expectations:

$$
\begin{align}
\nabla_\theta \log p(x; \theta) &= \nabla_\theta \log \sum_z \tilde{p}(x, z; \theta) - \nabla_\theta \log Z(\theta) \\
&= \mathbf{E}_{z \sim p(z \mid x)} \nabla_\theta \log \tilde{p}(x, z; \theta) - \mathbf{E}_{x, z \sim p(x, z)} \nabla_\theta \log \tilde{p}(x, z; \theta)
\end{align}
$$

which enables the use of Monte Carlo methods for maximizing the resulting
likelihood while avoiding intractability problems in both phases. This
maximization admits a useful interpretation of both phases. In the positive
phase, we increase $\log \tilde{p}(x, z; \theta)$ for $x$ drawn from the *data
distribution* (and $z$ sampled from the conditional). In the negative phase,
we decrease the partition function by decreasing $\log \tilde{p}(x, z)$ for
$x, z$ drawn from the *model distribution*. As a result, the positive phase
pushes *down* the energy of training examples and the negative phase pushes
*up* the energy of samples drawn from the model.

## Learning Undirected Models

Now that we've broken down the loss function corresponding to maximum likelihood
parameter estimation in undirected models into two parts, each written as
expectations over different distributions, we are prepared to discuss learning
algorithms for undirected models.

*A First Approach.* Since we've re-expressed our loss function as a sum of expectations, an
immediate and intuitive approach to approximating the loss is via Monte Carlo
approximations of each expectation. In particular, we can compute the positive
phase expectation by sampling $x$ from the training set, $z$ according to $p(z
\mid x)$, and computing the gradient with those $x$ and $z$.[^3] Since the negative
phase requires sampling from the model distribution, we can perform such
sampling by initializing $m$ samples of $(x, z)$ to random values and applying a
Markov Chain Monte Carlo method (such as Gibbs sampling or Metropolis-Hastings
sampling) to generate representative $(x, z)$ from the model distribution. The
expectation can then be computed as an average of $\nabla_\theta \tilde{p}(x, z;
\theta)$ over these samples. This approach is made explicit in Algorithm 1.

{% include pseudocode.html id="1" code="
\begin{algorithm}
\caption{A naive Markov Chain Monte Carlo algorithm for maximizing the log-likelihood of an undirected graphical model.}
\begin{algorithmic}

\PROCEDURE{NaiveMCMC}{k} \Comment{$k$ is the number of Gibbs steps}

    \While{convergence criteria are not met}
      \State{$g \gets 0$} 
      \State{Sample a minibatch of $m$ examples $\{x^{(1)} \dots x^{(m)}\}$ from the training set}
      \State{Sample $m$ examples $\{z^{(1)} \dots z^{(m)}\}$ where $z^{(i)} \sim p(z \mid x^{(i)})$}
      \State{$g \gets \frac{1}{m} \sum_{i=1}^m \nabla_\theta \tilde{p}(x^{(i)}, z^{(i)}; \theta)$}
      
      \State{Initialize a set of $m$ examples $\{\tilde{c}^{(1)} \dots \tilde{c}^{(m)}\}$ to random values} \Comment{Let $\tilde{c}^{(j)} = (\tilde{x}^{(j)}, \tilde{z}^{(j)})$}
      \For{$i = 1 \dots k$}
        \For{$j = 1 \dots m$}
          \State{$\tilde{c}^{(j)} \gets \texttt{gibbs\_update}(\tilde{c}^{(j)})$}
        \EndFor
      \EndFor
      \State{$g \gets g - \frac{1}{m} \sum_{i = 1}^m \nabla_\theta \log \tilde{p}(\tilde{x}^{(j)}, \tilde{z}^{(j)})$}
      \State{$\theta \gets \theta + \epsilon g$} \Comment{For some fixed $\epsilon$}
    \EndWhile
\ENDPROCEDURE
\end{algorithmic}
\end{algorithm}
" %}

While this naive method may work well in expectation, it has numerous issues in
practice that render it infeasible. In particular, the initialization of random
$x, z$ that are subsequently altered via repeated Gibbs sampling to reflect
samples from the model distribution is extremely sensitive to the randomly
chosen input and requires a burning in phase for each iteration of the
algorithm. A reasonable solution to this random initialization is to initialize
the Markov chains for the sampling procedure from a distribution that is close
to the model distribution; this will reduce the number of steps required for the
burn-in phase and reduce undesirable sensitivity to noise. Fortunately, we have
such a distribution that's easy to sample from on hand: the data distribution!
While the data distribution is quite distinct from the model distribution in
initial iterations, the distributions will ideally align as training proceeds,
after which the negative phase will start to become more accurate. 

*Contrastive Divergence.* The contrastive divergence (CD) algorithm does just
this, initializing the Markov chain at each step with samples from the data
distribution. As mentioned previously, this optimization is likely to perform
poorly at the beginning of the training process when the data and model
distributions are far apart, but as the postive phase is given time to act (and
improve the model's probability of the data), the model distribution will become
closer to the data distribution and therefore make the negative phase more
accurate. This approach is made explicit in Algorithm 2, where the only difference
from Algorithm 1 is the initialization of $\tilde{c}^{(j)}$ to the previously
sampled values from the data distribution (as opposed to random values).

{% include pseudocode.html id="2" code="
\begin{algorithm}
\caption{The contrastive divergence algorithm for maximizing the log-likelihood of an undirected graphical model.}
\begin{algorithmic}

\PROCEDURE{ContrastiveDivergence}{k} \Comment{$k$ is the number of Gibbs steps}

    \While{convergence criteria are not met}
      \State{$g \gets 0$} 
      \State{Sample a minibatch of $m$ examples $\{x^{(1)} \dots x^{(m)}\}$ from the training set}
      \State{Sample $m$ examples $\{z^{(1)} \dots z^{(m)}\}$ where $z^{(i)} \sim p(z \mid x^{(i)})$}
      \State{$g \gets \frac{1}{m} \sum_{i=1}^m \nabla_\theta \tilde{p}(x^{(i)}, z^{(i)}; \theta)$}
      
      \State{Initialize a set of $m$ examples $\{\tilde{c}^{(1)} \dots \tilde{c}^{(m)}\}$ to the previously sampled values} \Comment{Let $\tilde{c}^{(j)} = (\tilde{x}^{(j)}, \tilde{z}^{(j)})$}
      \For{$i = 1 \dots k$}
        \For{$j = 1 \dots m$}
          \State{$\tilde{c}^{(j)} \gets \texttt{gibbs\_update}(\tilde{c}^{(j)})$}
        \EndFor
      \EndFor
      \State{$g \gets g - \frac{1}{m} \sum_{i = 1}^m \nabla_\theta \log \tilde{p}(\tilde{x}^{(j)}, \tilde{z}^{(j)})$}
      \State{$\theta \gets \theta + \epsilon g$} \Comment{For some fixed $\epsilon$}
    \EndWhile
\ENDPROCEDURE
\end{algorithmic}
\end{algorithm}
" %}

In essence, contrastive divergence provides a reasonable approximation to the
negative phase, empirically performing better than the naive MCMC method
presented in Algorithm 1. However, the CD approximation isn't perfect --
specifically, it doesn't suppress regions of high probability under the model
that have low probability under the data distribution. Due to its initialization
of its Markov chains from data points, CD is unlikely to visit such regions
(called spurious modes) that are far from the data generating distribution. As a
result, models trained with CD may waste probability mass on these modes,
struggling to place high mass on the correct modes of the data generating
distribution. 

A variation on contrastive divergence that has seen some empirical success is
the stochastic maximum likelihood (also called persistent contrastive
divergence) algorithm, which is inspired by the idea that so long as the steps
taken by the gradient ascent algorithm are small, the model distribution will
not change substantially at each step {% cite swersky2010tutorial %}. As a result, samples from the previous
step will likely be close to being reasonable samples for the current model's
distribution, and as a result will require less Markov chain mixing.
Initializing Markov chains with samples from previous steps also allows these
chains to be continually updated through the learning process (as opposed to
being restarted with random values at each step), thereby increasing the
likelihood of exploring the whole search space and avoiding the spurious
mode problem present in CD. 

## Restricted Boltzmann Machines

We've spent the past two sections discussing the theoretical training of
undirected models; in this section, we'll discuss an undirected model, the
restricted boltzmann machine (RBM), that satisfies many of the desirable
properties that enable the use of algorithms such as contrastive divergence to
tackle the existence of a partition function {% cite hinton2012practical %}. In
particular, the RBM has a very desirable factorization allowing for the simple
computation of $p(z \mid x)$ and $p(x \mid z)$, which makes the Gibbs sampling
steps tractable.

The restricted boltzmann machine is defined over a set of visible variables $x$
and hidden variables $z$ so that

$$
p(x, z; \theta) = \frac{\exp(-\mathcal{E}(x, z; \theta))}{Z(\theta)}
$$

for energy function $\mathcal{E}$ defined over $D$-dimensional binary $x$ and $K$-dimensional binary $z$ as

$$
\mathcal{E}(x, z; \theta) = -\sum_{i = 1}^D \sum_{j = 1}^K v_i W_{ij} h_j - \sum_{i = 1}^D x_ic_i - \sum_{j = 1}^K z_jb_j
$$

where $\theta = (W, b, c)$. The graphical representation of RBMs with
connections solely between hidden and observed units (and no connections between
units of the same type) admits the property that all observed units become
independent when conditioned on hidden units (and vice versa). As derived in {%
cite ping2016learning %}, we have

$$
\begin{align*}
p(x_i = 1 \mid z, \theta) &= \sigma \left( \sum_{j = 1}^K W_{ij} z_j \right) \\
p(z_i = 1 \mid x, \theta) &= \sigma \left( \sum_{j = 1}^H W_{ij} x_j \right)
\end{align*}
$$

The simple forms of these expressions allow for a simple and efficient Gibbs
sampler. In particular, the RBM can be trained directly with Algorithm 2 (CD);
$p(z \mid x)$ in Line 5 is tractable by block sampling with $x$ fixed, and the
Gibbs update in Line 10 is tractable by repeated block sampling with $x$ fixed
to obtain an updated $z$ and $z$ fixed to obtain an updated $x$. By extension,
stochastic maximum likelihood can be used to train RBMs efficiently as well.
More discussion on RBM-specific training can be found in {% cite
swersky2010tutorial %}.

## Summary & Further Reading

Undirected graphical models allow for the more general expression of
dependencies between nodes through the use of unnormalized energy functions over
cliques of nodes. However, this increased expressivity comes at a cost when
learning such models: dealing with the intractable partition function when
maximizing model likelihood is challenging. 

In this post, we've shown that breaking down the likelihood function of
generalized undirected models into a positive and negative phase and
subsequently Monte Carlo estimating each phrase yields promising algorithms
for the optimization of such models. Varying initializations for the Gibbs
samplers to estimate both phases yield differing algorithms: the simple
Markov Chain Monte Carlo method employs a random initializer, the contrastive
divergence algorithm initializes Gibbs samplers with samples from the data
distribution, and stochastic maximum likelihood initializes samples from
those obtained at previous iterations (and performs best in practice). We
concluded with a brief foray into restricted boltzmann machines, shallow
undirected models that enable efficient training via the aformentioned
approaches due to their desirable factorization. 

Multiple other methods have been developed to train undirected models; the
Monte Carlo estimation procedure here is only one of many. In particular,
pseudolikelihood, score matching, and noise-contrastive estimation follow
different paradigms to deal with the partition function, and are worth
further examination. 

## Notes

[^1]: This is because the introduction of latent variables introduces a sum $p(x) = \sum_z p(x, z)$, which poses challenges when computing the log-likelihood of the data.
[^2]: If $p(z \mid x)$ is intractable, we can still derive a lower bound for the expectation; see [this post on latent variable model learning](https://mananshah99.github.io/blog/2020/06/23/lvm/) for further information.
[^3]: In this post, we assume that computing the conditional $p(z \mid x)$ is tractable, therefore allowing for a Monte Carlo evaluation of the expectation in the positive phase. If this were not the case, variational methods could be used to approximately evaluate the conditional, but we won't worry ourselves with those complexities here (see [this post](https://mananshah99.github.io/blog/2020/06/23/lvm/) if you're curious). 
