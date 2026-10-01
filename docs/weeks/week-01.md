# Week 1: Autograd & PyTorch from Zero

*Training notes, week 1. Lecture-style write-ups: definitions, worked derivations, intuition checks, exercises.*

## How to use these notes

- **Time budget:** ~2 hours/day, 5 days. Each lecture below is one day.
- **Rule 1:** Cover the worked examples and do them on paper *before* reading the steps.
- **Rule 2:** Exercises marked ★ are the interview bar — be able to do them cold, on a whiteboard, without notes.
- **Rule 3:** The end-of-week quiz is closed-book except where noted. Worked solutions stay private — don't peek until you've attempted the quiz.
- **Notation:** scalars in italics ($x$), vectors bold ($\mathbf{x}$), matrices bold capitals ($\mathbf{W}$). $L$ is always the loss. $\sigma$ is the sigmoid.

---

## Lecture 1 — The calculus underneath

*Day 1. Everything this week — and a disturbing fraction of ML interviews — is the chain rule, applied mechanically. Rebuild it until it's reflex.*

### 1.1 The derivative, in one paragraph

The derivative $f'(x)$ is the slope of $f$ at $x$: how much $f$ changes per unit change in $x$. Formally,

$$f'(x) = \lim_{h\to 0} \frac{f(x+h) - f(x)}{h}.$$

You will never compute this limit directly again after today. What matters is the *habit*: whenever you see a composition of functions, your reflex should be "chain rule," and whenever you see a sum of terms, "differentiate term by term." That reflex is 90% of backprop.

### 1.2 Six derivatives to know cold

| $f(x)$ | $f'(x)$ | Where you'll meet it |
|---|---|---|
| $x^2$ | $2x$ | every quadratic loss |
| $e^x$ | $e^x$ | softmax, exp ops |
| $\log x$ | $1/x$ | cross-entropy |
| $\sigma(x) = 1/(1+e^{-x})$ | $\sigma(x)(1-\sigma(x))$ | sigmoid activations, logistic regression |
| $\tanh(x)$ | $1 - \tanh^2(x)$ | tanh activations, RNNs (Week 2) |
| $\mathrm{ReLU}(x) = \max(0,x)$ | $1$ if $x>0$ else $0$ | every modern network |

**Worked example — derive the sigmoid derivative.** This is worth doing once by hand because the trick (expressing the derivative *in terms of the function itself*) recurs everywhere.

$\sigma(x) = (1 + e^{-x})^{-1}.$ By the chain rule:

$$\sigma'(x) = -(1 + e^{-x})^{-2} \cdot (-e^{-x}) = \frac{e^{-x}}{(1 + e^{-x})^2}.$$

Now split the fraction:

$$= \frac{1}{1+e^{-x}} \cdot \frac{e^{-x}}{1+e^{-x}} = \sigma(x) \cdot (1 - \sigma(x)),$$

since $\frac{e^{-x}}{1+e^{-x}} = \frac{1+e^{-x}-1}{1+e^{-x}} = 1 - \sigma(x)$. ∎

Why this matters practically: during backprop you already *have* $\sigma(x)$ stored from the forward pass, so the backward pass is one multiplication. Every `_backward` you'll write exploits something like this.

**tanh sketch:** $\tanh(x) = \sinh(x)/\cosh(x)$. Quotient rule gives $(\cosh^2 - \sinh^2)/\cosh^2 = 1/\cosh^2 = 1 - \tanh^2(x)$. Same moral: derivative expressed via the already-computed value.

**ReLU note:** ReLU isn't differentiable at $x = 0$. In practice everyone defines the subgradient there as 0 (or 1); it doesn't matter for training. It *does* matter for gradient checking (Lecture 3) — remember this.

### 1.3 Partial derivatives and the gradient

For $f(x, y)$, the partial derivative $\partial f/\partial x$ is the slope holding $y$ fixed — "pretend $y$ is a constant, differentiate normally." The **gradient** $\nabla f$ is the vector of all partials: it points in the direction of steepest increase, which is why training *descends* its negative.

**Worked example.** $f(x,y) = x^2y + \sin(y)$. Then $\partial f/\partial x = 2xy$ and $\partial f/\partial y = x^2 + \cos(y)$. At $(1, 0)$: $\nabla f = (0, 2)$.

### 1.4 The chain rule

**Single variable.** If $y = g(u)$ and $u = h(x)$, then

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}.$$

Derivatives multiply along the chain. That's it. That's backprop.

**Worked example 1.** $\frac{d}{dx}\left[e^{-x^2}\right]$. Outer: $\frac{d}{du} e^u = e^u$; inner: $\frac{d}{dx}(-x^2) = -2x$. Multiply: $-2x\cdot e^{-x^2}$.

**Worked example 2 — a preview of coming attractions.** $\frac{d}{dx}\left[\log(1 + e^x)\right]$. Outer: $1/(1+e^x)$; inner: $e^x$. Multiply: $\frac{e^x}{1+e^x} = \sigma(x)$. So the derivative of softplus is the sigmoid. File this away — compositions of these six functions keep collapsing into each other, which is why the table in §1.2 is short.

**Multivariable.** If $z = f(x, y)$ with $x = g(t)$, $y = h(t)$, then

$$\frac{dz}{dt} = \frac{\partial z}{\partial x}\frac{dx}{dt} + \frac{\partial z}{\partial y}\frac{dy}{dt}.$$

Every path from $t$ to $z$ contributes; you sum over paths. (When a node has multiple children in a computation graph, its gradient is the *sum* of contributions — same idea.)

**Worked example 3.** $z = x^2y + \sin(y)$, $x = e^t$, $y = t^3$. Find $dz/dt$ at $t = 1$.
$\partial z/\partial x = 2xy$, $\partial z/\partial y = x^2 + \cos(y)$, $dx/dt = e^t$, $dy/dt = 3t^2$.
At $t = 1$: $x = e$, $y = 1$.
$dz/dt = (2\cdot e\cdot 1)(e) + (e^2 + \cos 1)(3) = 2e^2 + 3e^2 + 3\cos(1) = 5e^2 + 3\cos(1) \approx 38.6.$

### 1.5 The sentence that is backprop

Memorize this phrasing — you'll say it in interviews:

> "Given how much the loss cares about my output, how much does it care about my inputs?"

Every `_backward` closure you write this week is that sentence, turned into code. Reverse-mode AD is just asking it once per node, in the right order.

### Exercises — Lecture 1

- **1.1** ★ From memory, on paper: derivatives of $x^2, e^x, \log x, \sigma(x), \tanh(x), \mathrm{ReLU}(x)$.
- **1.2** Compute $\frac{d}{dx}[\sin(x^2)]$, $\frac{d}{dx}[(x^3+1)^5]$, $\frac{d}{dx}[e^{-(x-1)^2}]$.
- **1.3** ★ Derive $\sigma'(x) = \sigma(x)(1-\sigma(x))$ without looking at §1.2.
- **1.4** If $z = f(x,y)$, $x = g(t)$, $y = h(t)$, write $dz/dt$ in full, then evaluate for $f = xy^2$, $x = \cos t$, $y = \sin t$ at $t = \pi/4$.
- **1.5** ★ Explain to a rubber duck: why does the multivariable chain rule *sum* over paths?
- **1.6** Show that $\frac{d}{dx}[\tanh(x)] = 1 - \tanh^2(x)$ using the quotient rule.

---
## Lecture 2 — Automatic differentiation: why reverse mode wins

*Day 2 (part A — the idea; part B is building it). Read this before you write any code.*

### 2.1 Three ways to get a derivative

1. **By hand** (what you did in Lecture 1). Exact, but doesn't scale past toy functions.
2. **Finite differences:** $f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}$. Easy, but needs $2n$ function evaluations for $n$ inputs, and $h$ is a trap: too big → truncation error, too small → floating-point roundoff eats you.
3. **Automatic differentiation:** apply the chain rule mechanically to the program's own operations. Exact (to floating point), fast. This is what every framework does.

### 2.2 Computation graphs

A program computing $f$ is a directed acyclic graph: **nodes** are intermediate values, **edges** are operations. Example:

$$f(x_1, x_2) = x_1\cdot x_2 + \sin(x_1)$$

Graph: $x_1, x_2 \to a = x_1\cdot x_2$; $x_1 \to b = \sin(x_1)$; $a, b \to f = a + b$.

The forward pass evaluates nodes in topological order. The backward pass walks the same graph in reverse, multiplying local derivatives — the chain rule, mechanized.

### 2.3 Forward mode

Carry pairs $(\text{value}, \text{derivative})$ through the program. **Seed** one input with derivative 1 and the rest with 0; the output's derivative slot then holds $\partial f/\partial x_i$. Cost: **one full pass per input**.

On our example at $(x_1, x_2) = (2, 3)$: pass 1 seeds $(\dot{x}_1, \dot{x}_2) = (1, 0)$ and yields $\partial f/\partial x_1 = x_2 + \cos(x_1) = 3 + \cos(2)$; pass 2 seeds $(0, 1)$ and yields $\partial f/\partial x_2 = x_1 = 2$. Two passes for two inputs.

### 2.4 Reverse mode

Define the **adjoint** $\bar{a} = \partial L/\partial a$: "how much the final output cares about $a$." Seed the *output* with adjoint 1, then sweep backward: each node distributes its adjoint to its parents via the local chain rule. Cost: **one full pass total**, and every input's gradient falls out.

On our example: $\bar{f} = 1 \to \bar{a} = 1$, $\bar{b} = 1 \to \bar{x}_2 = \bar{a}\cdot x_1 = 2$, and $\bar{x}_1 = \bar{a}\cdot x_2 + \bar{b}\cdot\cos(x_1) = 3 + \cos(2)$. One pass, both gradients.

### 2.5 The cost argument ★

$n$ inputs, $m$ outputs. Forward mode: $n$ passes. Reverse mode: $m$ passes. A neural network has $n = \text{millions of parameters}$ and $m = 1$ (the scalar loss). **Reverse mode wins by a factor of millions.** This two-sentence argument is a top-10 interview answer; own it.

### 2.6 The `Value` object — design before code

Tomorrow you implement this, but fix the design in your head tonight:

- **Fields:** `data` (the number), `grad` (its adjoint, init 0), `_prev` (parent nodes), `_op` (which operation made it, for debugging), `_backward` (a closure that pushes `grad` to parents).
- **Invariant:** a node's `_backward` may only run after *all* of its children's contributions to its `grad` are final. §3 explains how topological order guarantees this.
- **Each op's `_backward` is one line of calculus.** Multiplication $z = x\cdot y$: $\bar{x} \mathrel{+}= y\cdot\bar{z}$, $\bar{y} \mathrel{+}= x\cdot\bar{z}$. That's the whole "engine."

### Exercises — Lecture 2

- **2.1** ★ Say out loud, whiteboard-style: "Why reverse-mode and not forward-mode for neural networks?" (Two sentences. No notes.)
- **2.2** For $f(x_1,x_2,x_3) = x_1x_2x_3$, count forward-mode vs reverse-mode passes needed for the full gradient.
- **2.3** Draw the computation graph of $L = (x\cdot y + \tanh(x))^2$. Label every intermediate node.
- **2.4** By hand, compute all adjoints for the graph in 2.3 at $(x, y) = (1, 2)$ using reverse mode. (You'll verify this against code tomorrow.)

---
## Lecture 3 — `backward()`, topological order, and gradient checking

*Day 2 (part B — the build). This lecture is meant to be read with an editor open.*

### 3.1 Why order matters

A node's `_backward` distributes its *current* `grad` to its parents. If some child hasn't contributed yet when the node fires, the parent's gradient is silently wrong — no error, just bad learning. The guarantee we need:

> **Process nodes in reverse topological order** — every node's `grad` is final before its `_backward` runs.

Topological order = parents before children. Reverse it = children before parents. That's the whole trick.

### 3.2 Topological sort via DFS

```python
topo = []
visited = set()
def build(v):
    if v not in visited:
        visited.add(v)
        for p in v._prev:
            build(p)
        topo.append(v)   # post-order: parents land before children
build(loss)
for v in reversed(topo): # children before parents
    v._backward()
```

Note the subtlety: a node shared by multiple paths (e.g. $x$ used twice) is visited once but accumulates gradient contributions from *every* child — which is exactly the multivariable chain rule's "sum over paths" from Lecture 1. If your implementation overwrites instead of accumulating (`=` instead of `+=`), shared nodes break. This is the #1 micrograd bug.

### 3.3 Deriving every `_backward` ★

For each op, differentiate the output w.r.t. each input, then multiply by the incoming adjoint (`out.grad`). Memorize this table — it *is* backprop:

| op: $out =$ | $\partial out/\partial x$ → code |
|---|---|
| $x + y$ | `x.grad += out.grad`, `y.grad += out.grad` |
| $x \cdot y$ | `x.grad += y.data * out.grad`, `y.grad += x.data * out.grad` |
| $x / y$ | `x.grad += (1/y.data) * out.grad`, `y.grad += (-x.data/y.data**2) * out.grad` |
| $x^{n}$ | `x.grad += n * x.data**(n-1) * out.grad` |
| $\exp(x)$ | `x.grad += out.data * out.grad` (since $\frac{d}{dx} e^x = e^x$, and `out.data` *is* $e^x$) |
| $\tanh(x)$ | `x.grad += (1 - out.data**2) * out.grad` |
| $\mathrm{relu}(x)$ | `x.grad += (out.data > 0) * out.grad` |

**Worked check — division.** $out = x/y = x\cdot y^{-1}$. $\partial out/\partial x = 1/y$; $\partial out/\partial y = -x/y^2$. Multiply each by `out.grad` per the chain rule. In practice implement division as `x * y**-1` and get it for free — fewer hand derivations, fewer bugs.

### 3.4 Gradient checking: trust, but verify

Analytic gradients are easy to get subtly wrong. The check: compare against central differences,

$$f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}, \quad h = 10^{-5},$$

which has error $O(h^2)$. Use **relative error**:

$$\mathrm{rel\_err} = \frac{|\mathrm{analytic} - \mathrm{numeric}|}{\max(1,\, |\mathrm{analytic}| + |\mathrm{numeric}|)},$$

and demand $\mathrm{rel\_err} < 10^{-5}$ (float64; loosen to ~$10^{-3}$ in float32).

**The ReLU-at-zero trap.** At $x = 0$, central differences see the kink and report ≈ 0.5, while your analytic `_backward` returns 0 or 1. The check "fails" — correctly, because ReLU isn't differentiable there. This is expected, not a bug: gradient-check *away* from kinks, or use one-sided differences. If a check fails at a smooth point (say $\tanh$ at $x = 0.7$), your `_backward` is wrong. Period.

**Debugging protocol** when a check fails: (1) shrink to the smallest graph containing the op; (2) print analytic vs numeric per op; (3) re-derive the partial on paper — the bug is in the calculus, not the code, 90% of the time.

### Exercises — Lecture 3

- **3.1** ★ Write `_backward` for $out = x - y$ and $out = x^2$ from scratch.
- **3.2** In the DFS above, what breaks if you forget the `visited` set? (Hint: diamond graph $x \to a$, $x \to b$, $a,b \to out$.)
- **3.3** ★ Explain why `+=` and not `=` in every `_backward`. Give a concrete graph where `=` gives the wrong answer.
- **3.4** Gradient-check your $\tanh$ at $x \in \{-2, -0.5, 0, 0.7, 3\}$. Then do $\mathrm{relu}$ at $x \in \{-1, 0, 1\}$ and explain the $x = 0$ result in one sentence.

---
## Lecture 4 — From `Value`s to networks: the MLP

*Day 3. Today the engine learns.*

### 4.1 Neuron, Layer, MLP

A neuron is an affine map plus a nonlinearity: $out = \sigma(\sum_i w_i x_i + b)$. Without the nonlinearity, stacked layers collapse into one linear map (a product of matrices is a matrix) — depth would be pointless. This is *the* reason activations exist.

```python
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0.0)
    def __call__(self, x):
        return (sum(w*xi for w, xi in zip(self.w, x)) + self.b).tanh()
```

A `Layer` holds several neurons; an `MLP` chains layers. `mlp.parameters()` collects every `w` and `b` — the things `backward()` will differentiate.

### 4.2 Loss functions — and the most famous gradient in ML ★

**Regression:** mean squared error, $L = \frac{1}{n}\sum(\hat{y} - y)^2$.
**Classification:** softmax + cross-entropy. With logits $z$ and one-hot label $y$:

$$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}, \qquad L = -\sum_k y_k \log p_k.$$

**Theorem (learn this derivation cold).** $\partial L/\partial z_i = p_i - y_i.$

*Proof.* First, $\partial p_k/\partial z_i = p_k(\delta_{ki} - p_i)$: if $k = i$, quotient rule gives $p_i(1 - p_i)$; if $k \neq i$, the numerator is constant and you get $-p_k p_i$. Then

$$\frac{\partial L}{\partial z_i} = -\sum_k y_k \cdot \frac{1}{p_k} \cdot \frac{\partial p_k}{\partial z_i} = -\sum_k y_k(\delta_{ki} - p_i) = -y_i + p_i\sum_k y_k = p_i - y_i,$$

since $\sum_k y_k = 1$. ∎

Read that result: the gradient is just *how wrong the probabilities are*. This is why softmax+CE trains so well — the backward signal is beautifully simple.

### 4.3 The training loop

The loop you'll write for the rest of your career:

1. **Forward:** `loss = L(model(x), y)` — builds the graph.
2. **Backward:** zero grads, `loss.backward()` — fills every `.grad`.
3. **Update:** `p.data -= lr * p.grad` for each parameter (SGD).

Two classic bugs live here: forgetting to zero grads (they *accumulate* — Lecture 5 explains why PyTorch does this on purpose), and a learning rate that's too big (loss explodes/NaNs) or too small (loss crawls). If `make_moons` doesn't separate after a few hundred steps, check these before anything else.

### 4.4 The `make_moons` experiment

`sklearn.datasets.make_moons(n=100, noise=0.1)`: two interleaving crescents, not linearly separable — a linear model scores ~85%, a 2-layer MLP with tanh should clear ~95%+. Train, print loss every 50 steps, then plot the decision boundary (evaluate the model on a grid, contour-plot the sign). *Seeing* the boundary wrap around the moons is the moment the machinery becomes intuition.

### 4.5 Sanity checks that save hours

- **Overfit a single batch first.** If the model can't memorize 8 points, the bug is in the code, not the hyperparameters.
- **Loss should fall fast, then slow.** Flat from step 0 → lr too small or gradients not flowing (check a `_backward`). NaN → lr too big or exp overflow.
- **Compare against PyTorch** (Lecture 5) with identical init and lr — the loss curves should match to ~$10^{-5}$. If they don't, your engine has a bug, and now you know exactly where to look.

### Exercises — Lecture 4

- **4.1** ★ Prove: a 2-layer network with *no* nonlinearity is just a linear model (write $\mathbf{W}_2(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$ as $\mathbf{W}'\mathbf{x} + \mathbf{b}'$).
- **4.2** ★ Derive $\partial L/\partial z_i = p_i - y_i$ without looking at §4.2.
- **4.3** Your moons loss stalls at 0.69 (that's $-\log 0.5$ — random guessing). List three hypotheses in order of likelihood and how you'd test each.
- **4.4** Why does the decision-boundary plot use a *grid* of points rather than the training points?

---
## Lecture 5 — PyTorch autograd: the production version of your engine

*Day 4. Everything here maps 1:1 onto what you built — PyTorch just does it on tensors, in C++, with GPU support.*

### 5.1 Tensors, `requires_grad`, leaf vs non-leaf

`x = torch.tensor(2.0, requires_grad=True)` marks $x$ for tracking. Every op on $x$ records a `grad_fn` (PyTorch's name for your `_backward`) and builds the graph dynamically — **define-by-run**: the graph is rebuilt fresh each forward pass, unlike static-graph frameworks (JAX/XLA) that compile once.

- **Leaf node:** created by you with `requires_grad=True` (model parameters, inputs you differentiate). After `.backward()`, leaves have `.grad` populated.
- **Non-leaf node:** result of an op. Has `grad_fn`, but `.grad` stays `None` unless you call `.retain_grad()`. (Interviewers ask this.)

Mapping to your engine: `Value` ≈ non-leaf tensor; your input `Value`s ≈ leaves.

### 5.2 `.backward()` and accumulation ★

`loss.backward()` walks the graph in reverse topological order calling each `grad_fn` — exactly your Lecture 3 loop. One deliberate difference: gradients **accumulate** (`+=`) into `.grad` instead of overwriting. Why? So you can call `.backward()` on multiple losses (multi-task learning, gradient accumulation across micro-batches) and the grads sum correctly.

Consequence — the most common PyTorch bug in existence: **you must zero grads each step** (`optimizer.zero_grad()`), or yesterday's gradients pollute today's update. Symptom of forgetting: loss improves, then diverges or oscillates, and nothing in the model looks wrong.

### 5.3 `no_grad()` and inference

`with torch.no_grad():` disables graph construction — no `grad_fn`s, no memory held for backward. Use it for **evaluation and inference**: it's faster and uses far less memory. `detach()` is the tensor-level version: returns a tensor sharing storage but cut from the graph (use when you need a value out of the graph without a context manager, e.g. logging, target networks).

### 5.4 In-place ops and the version counter ★

Ops ending in `_` (`relu_()`, `add_()`) mutate the tensor. PyTorch tracks a **version counter** per tensor; the engine saved *specific tensor values* for backward, and an in-place change invalidates them. Hence the famous error:

> *"a leaf Variable that requires grad is being used in an in-place operation"*

Translation: you mutated a leaf after the graph referenced its old value, so backward would compute with corrupted saved tensors. Fix: don't do in-place ops on tensors needed for backward — use out-of-place versions.

### 5.5 Port your MLP and match results

Rebuild Lecture 4's MLP with `torch.nn`: `nn.Linear` layers, `tanh`, `CrossEntropyLoss`, manual SGD loop. With identical initialization and learning rate, your micrograd loss curve and PyTorch's should agree to ~$10^{-5}$. This agreement is the strongest evidence your engine is correct — two independent implementations of the same math converging.

### Exercises — Lecture 5

- **5.1** ★ After `loss.backward()`, which tensors have non-`None` `.grad`: leaves, non-leaves, or both? How do you get the gradient of a non-leaf?
- **5.2** ★ You forgot `zero_grad()`. Describe the training symptom and explain the mechanism.
- **5.3** When do you use `no_grad()` vs `detach()`? Give one concrete use case for each.
- **5.4** ★ Explain the in-place error message above as if to a teammate: what did they do, why does autograd care, what's the fix?

---

## Lecture 6 — Custom `autograd.Function` and the checkpoint derivation

*Day 5. The week ends where real ML engineering begins: ops PyTorch doesn't ship.*

### 6.1 Why custom ops exist

Novel activations, fused kernels, quantized ops, custom losses with hand-optimized backward passes — research and production both need operations the framework never heard of. `torch.autograd.Function` is the supported way to add one.

### 6.2 Anatomy

```python
class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)      # stash what backward needs
        return x * torch.sigmoid(x)
    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        s = torch.sigmoid(x)
        grad_x = grad_out * (s + x * s * (1 - s))   # hand-derived
        return grad_x
```

Rules: `forward`/`backward` are static; `ctx` carries saved tensors; `backward` receives the adjoint and returns one gradient per input. Then verify with `torch.autograd.gradcheck` — the framework's built-in version of your Lecture 3 checker.

**Worked derivation — Swish backward.** $f(x) = x\cdot\sigma(x)$. Product rule: $f' = \sigma(x) + x\cdot\sigma'(x) = \sigma(x) + x\cdot\sigma(x)(1-\sigma(x))$. That's the line in `backward` above.

### 6.3 ★ The checkpoint derivation: 2-layer backprop in matrix form

Do this on paper, without notes, before Sunday. Setup: $x \in \mathbb{R}^d$, hidden size $h$, classes $c$.

$$z_1 = \mathbf{W}_1 x + b_1, \quad a_1 = \sigma(z_1), \quad z_2 = \mathbf{W}_2 a_1 + b_2, \quad p = \mathrm{softmax}(z_2), \quad L = \mathrm{CE}(p, y).$$

Define error signals $\delta_2 = \partial L/\partial z_2$, $\delta_1 = \partial L/\partial z_1$. From Lecture 4, $\delta_2 = p - y$ (the famous result). Then:

- $\partial L/\partial \mathbf{W}_2 = \delta_2 a_1^\top$, $\partial L/\partial b_2 = \delta_2$ (each weight's gradient = error × the input that flowed through it)
- $\delta_1 = (\mathbf{W}_2^\top\delta_2) \odot \sigma'(z_1)$ — error flows back through $\mathbf{W}_2$, then through the nonlinearity elementwise
- $\partial L/\partial \mathbf{W}_1 = \delta_1 x^\top$, $\partial L/\partial b_1 = \delta_1$

Dimensions check (do this every time): $\delta_2$ is $c\times 1$, $a_1^\top$ is $1\times h$ → $\mathbf{W}_2$ grad is $c\times h$ ✓. $\mathbf{W}_2^\top\delta_2$ is $h\times 1$, $\odot\, \sigma'(z_1)$ keeps $h\times 1$ ✓. $\delta_1 x^\top$: $h\times 1$ times $1\times d$ → $h\times d$ ✓.

If you can write this from a blank page and justify every line, you own the single most-probed derivation in ML interviews.

### Exercises — Lecture 6

- **6.1** ★ Derive the Swish backward above without looking.
- **6.2** Implement a clipped-exponential op $f(x) = \min(e^x, 10)$ as an `autograd.Function`, hand-derive its backward (careful at the clip boundary), and gradcheck it.
- **6.3** ★★ Closed book: the full 2-layer derivation in §6.3, with the dimension check.

---

## Watch list — everything to watch, in order

1. **Karpathy — "Let's build micrograd"** ([YouTube](https://www.youtube.com/watch?v=VMj-3S1tku0)). Watch the first half on Day 1 (through the `Value` class), the second half on Day 2. Then close it and build from memory — the video is a scaffold, not the artifact.
2. **PyTorch — "A Gentle Introduction to `torch.autograd`"** ([pytorch.org](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)). Read on Day 4 alongside Lecture 5; sections on *Differentiation in Autograd* and *Computational Graph* map directly onto Lectures 2–3.
3. **Reference implementation:** `karpathy/micrograd` on GitHub — compare against yours *after* your engine passes gradient checks, not before.
4. Optional: Baydin et al., "Automatic Differentiation in Machine Learning: a Survey" — §2 (what AD is *not*) and §3.2 (reverse mode costs).

---

## Interview drill bank (whiteboard, out loud)

1. ★★ Derive backprop for a 2-layer MLP in vectorized form. (Do until boring.)
2. ★ "Why does `.grad` accumulate instead of overwrite? What breaks if you forget `zero_grad()`?"
3. ★ "Why reverse-mode and not forward-mode for neural networks?"
4. ★ "An in-place ReLU on a tensor needed for backward — what happens and why?"
5. ★ "Explain the difference between `torch.no_grad()` and `detach()`."
6. "Your gradient check fails only at ReLU(0). Bug or expected? Why?"
7. "Softmax + cross-entropy: what's $\partial L/\partial z$, and why is the simplicity significant?"

---

## Checkpoint — do not advance if shaky

From a blank file: implement `Value` + `backward()`, train the moons MLP, explain every line. On paper, no notes: derive 2-layer backprop (§6.3). If either is shaky, repeat the week — foundations compound, and so do gaps.

*End-of-week quiz: take it Sunday before the check-in. Solutions stay private — attempt first.*
