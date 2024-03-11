package com.np_solr;

public class Distance {

    Distance() {
    }

    /**
     * Gets the Kullback Leibler divergence.
     * 
     * @param p P vector.
     * @param q Q vector.
     * @return The Kullback Leibler divergence between u and v.
     */
    public double KullbackLeiblerDivergence(double[] p, double[] q) {

        double k = 0;

        for (int i = 0; i < p.length; i++) {
            if (p[i] != 0 && q[i] != 0) {
                k += p[i] * Math.log(p[i] / q[i]);
            }
        }
        return k;
    }

    /**
     * Gets the Jensen Shannon divergence.
     * 
     * @param p U vector.
     * @param q V vector.
     * @return The Jensen Shannon divergence between u and v.
     */
    public double JensenShannonDivergence(double[] p, double[] q) {

        // if (p.length != q.length) {
        // throw new IllegalArgumentException(
        // String.format("Arrays have different length: p[%d], q[%d]", p.length,
        // q.length));
        // }

        double[] m = new double[p.length];
        for (int i = 0; i < m.length; i++) {
            m[i] = (p[i] + q[i]) / 2;
        }

        return (KullbackLeiblerDivergence(p, m) + KullbackLeiblerDivergence(q, m)) / 2;
    }

    public double bhattacharyyaDistance(double[] p, double[] q) {
        // if (p.length != q.length) {
        // throw new IllegalArgumentException("Distributions must have the same
        // length");
        // }

        double sum = 0.0;
        for (int i = 0; i < p.length; i++) {
            sum += Math.sqrt(p[i] * q[i]);
        }

        return sum;
    }

    /**
     * Gets the Jensen Shannon divergence.
     * 
     * @param p U vector.
     * @param q V vector.
     * @return The Jensen Shannon divergence between u and v.
     */
    public double cosineDistance(double[] p, double[] q) {
        if (p.length != q.length) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }

        double dotProduct = 0.0;
        double pMagnitude = 0.0;
        double qMagnitude = 0.0;

        for (int i = 0; i < p.length; i++) {
            dotProduct += p[i] * q[i];
            pMagnitude += p[i] * p[i];
            qMagnitude += q[i] * q[i];
        }

        pMagnitude = Math.sqrt(pMagnitude);
        qMagnitude = Math.sqrt(qMagnitude);

        if (pMagnitude == 0 || qMagnitude == 0) {
            throw new IllegalArgumentException("One or both vectors have zero magnitude");
        }

        return dotProduct / (pMagnitude * qMagnitude);
    }

}
