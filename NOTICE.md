# Attribution and external data

GWIPG code is released under the MIT license in `LICENSE`. The license text
follows <https://opensource.org/license/mit>.

POT (Python Optimal Transport) supplies the GW objective, gradients, and
comparison solvers; it is an independently installed MIT-licensed dependency.
The low-rank curve example follows the construction in
<https://pythonot.github.io/auto_examples/others/plot_lowrank_GW.html>.

SNARE-seq inputs are third-party data distributed by SCOT at
<https://rsinghlab.github.io/SCOT/data/>. They are downloaded on demand and
excluded from source distributions and Git by default. The GWIPG MIT license
does not relicense those data. Cite Chen, Lake, and Zhang (2019),
doi:10.1038/s41587-019-0290-0, and Demetci et al. (2022),
doi:10.1089/cmb.2021.0446, when using that example.

The analytical sphere reference is Arya et al. (2026),
doi:10.1007/s10208-024-09678-3. The benchmark uses unnormalized squared
Euclidean costs and the continuous (4,2)-GW convention described in the
reproduction guide.
