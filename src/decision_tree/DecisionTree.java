package decision_tree;

public abstract class DecisionTree {
	public abstract void fit(double[][] X, double[] Y, double[] sample_weight);

	public abstract Node apply(double[] x);
}
