package objective;

public class QuantileEstimator extends Estimator {
	private double alpha;
	private double quantile;

	public QuantileEstimator(double alpha) {
		this.alpha = alpha;
	}

	public void fit(double[][] X, double[] y, double[] sample_weight) {
		this.quantile = Utils.weighted_percentile(y, sample_weight, this.alpha * 100.0);
		
	}

	public double[] predict(double[][] X) {
		int size = X.length;
		double[] ans = new double[size];
		for (int i = 0; i < size; i++)
			ans[i] = this.quantile;
		return ans;
	}
}