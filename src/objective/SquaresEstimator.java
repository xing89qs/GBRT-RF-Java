package objective;

import java.util.Arrays;

public class SquaresEstimator extends Estimator {

	private double mean;

	@Override
	public void fit(double[][] X, double[] y, double[] sample_weight) {
		// TODO Auto-generated method stub
		this.mean = Utils.averavge(y, sample_weight);
	}

	@Override
	public double[] predict(double[][] X) {
		double[] ans = new double[X.length];
		Arrays.fill(ans, mean);
		return ans;
	}

}
