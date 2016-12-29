package objective;

import decision_tree.DecisionTree;

public abstract class LossFunction {
	public abstract double[] negative_gradient(double[] y_true, double[] y_pred, double[] residual);

	public abstract void update_terminal_region(DecisionTree tree, double[][] sample, double[] label, double[] y_pred,
			double[] sample_weight);
}