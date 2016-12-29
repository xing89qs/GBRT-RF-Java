package objective;

import objective.Utils.Item;

public abstract class Estimator {
	public abstract void fit(double[][] X, double[] y, double[] sample_weight);

	public abstract double[] predict(double[][] X);
}