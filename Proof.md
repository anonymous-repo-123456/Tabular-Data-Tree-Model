# Convergence Analysis of the Block Splitting Process

In this section, we establish the convergence of the proposed block splitting algorithm. We prove that each split operation results in two non-empty subblocks, ensuring the process terminates.

## Assumptions

We make the following standard assumptions for the analysis:

**Assumption 1**
The block to be split contains at least two samples (\(n \geq 2\)). Furthermore, the block contains at least one sample belonging to the target class and at least one sample belonging to the non-target class.

**Assumption 2**
The feature-weighted sums \(\text{FS}(\boldsymbol{x}_j)\) are distinct for at least two samples within the block. This ensures that a split is always possible.

**Assumption 3**
The threshold \(\gamma\) used in the splitting criterion is a positive integer satisfying \(1 \leq \gamma \leq n\).

## Notation and Setup

Let a block consist of \(n\) samples \(\{\boldsymbol{x}_1, \ldots, \boldsymbol{x}_n\}\), where each \(\boldsymbol{x}_j \in \mathbb{R}^m\). The feature-weighted sum for sample \(\boldsymbol{x}_j\) is defined as:
\[
\text{FS}(\boldsymbol{x}_j) = \sum_{i=1}^m w_i x_{ij},
\]
where \(w_i\) represents the weight associated with the \(i\)-th feature. The block splitting process partitions the samples based on a threshold \(c\). A sample \(\boldsymbol{x}_j\) is assigned to the first subblock if \(\text{FS}(\boldsymbol{x}_j) < c\) and to the second subblock if \(\text{FS}(\boldsymbol{x}_j) \geq c\).

Let \(\mathcal{S} = \{1, \ldots, n\}\) be the index set of samples in the block. Let \(\mathcal{S}^\text{t} \subseteq \mathcal{S}\) be the index set of target class samples and \(\mathcal{S}^\text{nt} \subseteq \mathcal{S}\) be the index set of non-target class samples. By Assumption 1, \(\mathcal{S}^\text{t} \neq \emptyset\) and \(\mathcal{S}^\text{nt} \neq \emptyset\). Let \(\mathcal{FS} = \{\text{FS}(\boldsymbol{x}_j) \mid j \in \mathcal{S}\}\) be the set of feature-weighted sums. We define:

*   \(\mathcal{FS}^\text{t} = \{\text{FS}(\boldsymbol{x}_j) \mid j \in \mathcal{S}^\text{t}\}\)
*   \(\mathcal{FS}^\text{nt} = \{\text{FS}(\boldsymbol{x}_j) \mid j \in \mathcal{S}^\text{nt}\}\)
*   \(\max \text{TFS} = \max(\mathcal{FS}^\text{t})\), \(\min \text{TFS} = \min(\mathcal{FS}^\text{t})\)
*   \(\max \text{NFS} = \max(\mathcal{FS}^\text{nt})\), \(\min \text{NFS} = \min(\mathcal{FS}^\text{nt})\)

Note that these extrema exist since \(\mathcal{FS}^\text{t}\) and \(\mathcal{FS}^\text{nt}\) are non-empty finite sets.

The splitting threshold \(c\) is selected based on the following quantities:
\[
\begin{aligned}
N_1 &= | \{ j \in \mathcal{S}^\text{t} \mid \text{FS}(\boldsymbol{x}_j) < \min \text{NFS} \} |, \\
N_2 &= | \{ j \in \mathcal{S}^\text{t} \mid \text{FS}(\boldsymbol{x}_j) > \max \text{NFS} \} |, \\
N_3 &= | \{ j \in \mathcal{S}^\text{nt} \mid \text{FS}(\boldsymbol{x}_j) < \min \text{TFS} \} |, \\
N_4 &= | \{ j \in \mathcal{S}^\text{nt} \mid \text{FS}(\boldsymbol{x}_j) > \max \text{TFS} \} |.
\end{aligned}
\]
Let \(N_{\max} = \max\{N_1, N_2, N_3, N_4\}\). The threshold \(c\) is chosen as follows:
\[
c = \begin{cases}
\min \text{NFS}, & \text{if } N_1 = N_{\max} \text{ and } N_{\max} \geq \gamma, \\
\max \text{NFS} + \delta_1, & \text{if } N_2 = N_{\max} \text{ and } N_{\max} \geq \gamma, \\
\min \text{TFS}, & \text{if } N_3 = N_{\max} \text{ and } N_{\max} \geq \gamma, \\
\max \text{TFS} + \delta_1, & \text{if } N_4 = N_{\max} \text{ and } N_{\max} \geq \gamma, \\
e, & \text{if } N_{\max} < \gamma,
\end{cases}
\]
where \(e = \frac{\min \text{NFS} + \max \text{NFS} + \min \text{TFS} + \max \text{TFS}}{4}\), and \(\delta_1\) is an infinitesimally small positive number ensuring that \(c\) is strictly greater than \(\max \text{NFS}\) or \(\max \text{TFS}\) respectively, but smaller than the next distinct value in \(\mathcal{FS}\) if one exists.

## Convergence Guarantee

We now state and prove the main theorem regarding the convergence of the splitting process.

**Theorem 1**
Under Assumptions 1, 2, and 3, the block splitting process based on the threshold \(c\) defined above always partitions the block into two non-empty subblocks: \(B_1 = \{\boldsymbol{x}_j \mid \text{FS}(\boldsymbol{x}_j) < c\}\) and \(B_2 = \{\boldsymbol{x}_j \mid \text{FS}(\boldsymbol{x}_j) \geq c\}\).

**Proof**
We prove the theorem by considering two cases based on the value of \(c\). Let \(B_1 = \{\boldsymbol{x}_j \mid \text{FS}(\boldsymbol{x}_j) < c\}\) and \(B_2 = \{\boldsymbol{x}_j \mid \text{FS}(\boldsymbol{x}_j) \geq c\}\). We need to show that \(B_1 \neq \emptyset\) and \(B_2 \neq \emptyset\).

**Case 1: \( c \neq e \)**
In this case, \(N_{\max} \geq \gamma \geq 1\). This implies that at least one of \(N_1, N_2, N_3, N_4\) is positive. We analyze the sub-cases based on the choice of \(c\):

*   **Sub-case 1.1: \(c = \min \text{NFS}\)**.
    Here, \(N_1 = N_{\max} \geq \gamma\). By definition, there are \(N_1 \geq 1\) target samples \(\boldsymbol{x}_j\) such that \(\text{FS}(\boldsymbol{x}_j) < \min \text{NFS} = c\). These samples belong to \(B_1\). Thus, \(B_1 \neq \emptyset\).
    Furthermore, all non-target samples \(\boldsymbol{x}_k\) satisfy \(\text{FS}(\boldsymbol{x}_k) \geq \min \text{NFS} = c\). Since the original block contains at least one non-target sample (Assumption 1), these non-target samples belong to \(B_2\). Thus, \(B_2 \neq \emptyset\).

*   **Sub-case 1.2: \(c = \max \text{NFS} + \delta_1\)**.
    Here, \(N_2 = N_{\max} \geq \gamma\). By definition, there are \(N_2 \geq 1\) target samples \(\boldsymbol{x}_j\) such that \(\text{FS}(\boldsymbol{x}_j) > \max \text{NFS}\). Since \(\delta_1\) is infinitesimally small, \(\text{FS}(\boldsymbol{x}_j) \geq \max \text{NFS} + \delta_1 = c\) holds for these samples (assuming no \(\text{FS}(\boldsymbol{x}_j)\) falls exactly at \(\max \text{NFS}\); if it does, the strict inequality \(>\) in the definition of \(N_2\) applies. The use of \(\delta_1\) ensures the split occurs correctly even with potential ties if handled properly, but the core idea is that these \(N_2\) samples end up in \(B_2\)). Thus, \(B_2 \neq \emptyset\).
    All non-target samples \(\boldsymbol{x}_k\) satisfy \(\text{FS}(\boldsymbol{x}_k) \leq \max \text{NFS} < \max \text{NFS} + \delta_1 = c\). Since the block contains at least one non-target sample, these samples belong to \(B_1\). Thus, \(B_1 \neq \emptyset\).

*   **Sub-case 1.3: \(c = \min \text{TFS}\)**.
    Here, \(N_3 = N_{\max} \geq \gamma\). By definition, there are \(N_3 \geq 1\) non-target samples \(\boldsymbol{x}_j\) such that \(\text{FS}(\boldsymbol{x}_j) < \min \text{TFS} = c\). These samples belong to \(B_1\). Thus, \(B_1 \neq \emptyset\).
    All target samples \(\boldsymbol{x}_k\) satisfy \(\text{FS}(\boldsymbol{x}_k) \geq \min \text{TFS} = c\). Since the original block contains at least one target sample (Assumption 1), these target samples belong to \(B_2\). Thus, \(B_2 \neq \emptyset\).

*   **Sub-case 1.4: \(c = \max \text{TFS} + \delta_1\)**.
    Here, \(N_4 = N_{\max} \geq \gamma\). By definition, there are \(N_4 \geq 1\) non-target samples \(\boldsymbol{x}_j\) such that \(\text{FS}(\boldsymbol{x}_j) > \max \text{TFS}\). As in Sub-case 1.2, these samples satisfy \(\text{FS}(\boldsymbol{x}_j) \geq \max \text{TFS} + \delta_1 = c\) and belong to \(B_2\). Thus, \(B_2 \neq \emptyset\).
    All target samples \(\boldsymbol{x}_k\) satisfy \(\text{FS}(\boldsymbol{x}_k) \leq \max \text{TFS} < \max \text{TFS} + \delta_1 = c\). Since the block contains at least one target sample, these samples belong to \(B_1\). Thus, \(B_1 \neq \emptyset\).

In all sub-cases where \(c \neq e\), both \(B_1\) and \(B_2\) are non-empty.

**Case 2: \( c = e \)**
This case occurs when \(N_{\max} < \gamma\). The threshold \(c = e\) is the average of the four extreme feature-weighted sum values (\(\min \text{NFS}, \max \text{NFS}, \min \text{TFS}, \max \text{TFS}\)). Let \(\text{FS}_{\min} = \min(\mathcal{FS})\) and \(\text{FS}_{\max} = \max(\mathcal{FS})\). By definition, \(\min \text{NFS} \geq \text{FS}_{\min}\), \(\max \text{NFS} \leq \text{FS}_{\max}\), \(\min \text{TFS} \geq \text{FS}_{\min}\), and \(\max \text{TFS} \leq \text{FS}_{\max}\). Therefore,
\[
\text{FS}_{\min} \leq \frac{\text{FS}_{\min} + \text{FS}_{\min} + \text{FS}_{\min} + \text{FS}_{\min}}{4} \leq e \leq \frac{\text{FS}_{\max} + \text{FS}_{\max} + \text{FS}_{\max} + \text{FS}_{\max}}{4} = \text{FS}_{\max}.
\]
So, \(e\) lies within the range \([\text{FS}_{\min}, \text{FS}_{\max}]\).
By Assumption 2, there exist at least two samples with distinct feature-weighted sums. Thus, \(\text{FS}_{\min} < \text{FS}_{\max}\) (since \(n \geq 2\)).
If all \(\text{FS}(\boldsymbol{x}_j)\) were equal, the block would not be splittable, contradicting Assumption 2.
Therefore, there must be at least one sample \(\boldsymbol{x}_k\) such that \(\text{FS}(\boldsymbol{x}_k) = \text{FS}_{\min}\) and at least one sample \(\boldsymbol{x}_l\) such that \(\text{FS}(\boldsymbol{x}_l) = \text{FS}_{\max}\).

Can \(c = e\) be equal to \(\text{FS}_{\min}\) or \(\text{FS}_{\max}\)?
If \(e = \text{FS}_{\min}\), then \(\min \text{NFS} = \max \text{NFS} = \min \text{TFS} = \max \text{TFS} = \text{FS}_{\min}\). This implies all samples have the same feature-weighted sum \(\text{FS}_{\min}\), contradicting Assumption 2.
Similarly, \(e = \text{FS}_{\max}\) leads to a contradiction.
Therefore, \(\text{FS}_{\min} < e < \text{FS}_{\max}\).

Since \(\text{FS}_{\min} < e\), the sample(s) \(\boldsymbol{x}_k\) with \(\text{FS}(\boldsymbol{x}_k) = \text{FS}_{\min}\) satisfy \(\text{FS}(\boldsymbol{x}_k) < e\). Thus, \(B_1 \neq \emptyset\).
Since \(e < \text{FS}_{\max}\), the sample(s) \(\boldsymbol{x}_l\) with \(\text{FS}(\boldsymbol{x}_l) = \text{FS}_{\max}\) satisfy \(\text{FS}(\boldsymbol{x}_l) > e\), which implies \(\text{FS}(\boldsymbol{x}_l) \geq e\). Thus, \(B_2 \neq \emptyset\).

In both Case 1 and Case 2, the splitting process results in two non-empty subblocks \(B_1\) and \(B_2\). Since each split reduces the size of the block being considered (as \(|B_1| \geq 1\), \(|B_2| \geq 1\), and \(|B_1| + |B_2| = n\)), and the process stops when blocks meet certain criteria (e.g., size or purity), the overall block splitting process is guaranteed to converge.
