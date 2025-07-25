# Vectorized MES
Consider $N$ agents and $M$ candidates (projects)
At moment $t$ we have:
* $B_t\in\mathbb{R}^N_{\geq0}$: vector with budgets/candidate left
* $U\in\mathbb{R}^{N\times M}_{\geq0}$: table of utilities
* * by $u_j$ we will denote utilities for the j-th project

Then we compute helper variables:
* $t_j=B_t\div u_j$. The value under $t_j[i]$ is the max $\rho$-value that voter $i$ is able to pay when contributing equally to project $j$
* let $t_j'$ be $t_j$ sorted in non-decreasing order. $\sigma_j$ is permutation st. $t_j'[i]=t_j[\sigma_j(i)]$
* * the following calculations rely on the observation that the candidates that would spend their entire budget on $j$ have associated values as a consistent prefix of $t_j'$
* now $B_j'$ will be vector st. $B_j'[i]=B[\sigma_j(i)]$
* $P_j$ is vector of prefix sums of $B_j'$. More precisely, $P_j[i]=\sum_{k=0}^{k\leq i - 1}B_j'[k]$. The value under index $i$ in $P_j$ can be viewed as the combined budget of candidates that are spending their entire budged on project $j$ assuming the $\rho$ between $t_j'[i-1]$ and $t_j'[i]$ (for convenience we assume $t_j[-1] = 0$)
* Now let $u_j'$ be permutated vector of utilities regarding project $j$ st $u_j'[i]=u_j[\sigma_j(i)]$.
* Now $S_j$ will be a suffix sum of $u_j'$. This means that $S_j[i]=\sum_{k=i}^Nu_j'[i]$. The value under $i$ in such vector is the sum of utilities of agents who are able to contribute "equally" to purchasement of project $j$ when $\rho$ value is between $t_j'[i-1]$ and $t_j'[i]$
* Now we compute $a_j=S_j * t_j'$ (element-wise multiplication). 
* Finally, we compute $b_j=a_j+P_j$

The next steps consist of:
* For a given project $j$ we take the smallest $i$ st. $b_j[i]\geq 1$ (assuming unit costs). 
* We find $rho$ st. $1-P_j[i-1]=S_j[i-1]*\rho$ This is the lowest $\rho$ value st. $j$ is $\rho$-affordable.
* Having those $\rho$s, we choose the next project, update the current budget vector and proceed to the next step. Optionally we exclude the unfeasible projects.