===================
Binding Motif Model
===================

Starting with a `.narrowPeak` file and a bam file , e filter all reads that starts within a window of `w` around each peak summit. We assume that a proportion `p_s` of the reads are there "because" they had a protein attached to it, and `1-p_s` proportion of the reads are randomly distriuted across the genome. The binding position `Z` of proteins in a given window is assumed to be multinomially distributed, where the parameters `theta` correspoind to binding-affinity. Finally, the distance from the binding position of the protein to the start of the read is assumed to be Geometric/Negative-Binomially distributed.  And the strand is uniformly drawn betweeen '+', '-'.

Thus the probability of observing a read `(strand, position)`:

.. math::

   Z \sim Multinomial(\theta)

   \delta \sim Geometric(\eta)

   B \sim Bernoullli(p_s)

   P(D, X | X \in W) = P(\tilde{B})/w + P(B)P(Z+\delta=X)/2
