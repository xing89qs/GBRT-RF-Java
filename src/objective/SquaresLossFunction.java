package objective;

import java.util.HashSet;

import decision_tree.DecisionRegressionTree;
import decision_tree.DecisionTree;
import decision_tree.Node;

public class SquaresLossFunction extends LossFunction {

	public SquaresLossFunction() {
		super();
		// TODO Auto-generated constructor stub
	}

	@Override
	public double[] negative_gradient(double[] y_true, double[] y_pred, double[] residual) {
		// TODO Auto-generated method stub
		for (int i = 0; i < y_true.length; i++)
			residual[i] = y_true[i] - y_pred[i];
		return residual;
	}

	@Override
	public void update_terminal_region(DecisionTree tree, double[][] x, double[] y, double[] y_pred,
			double[] sample_weight) {
		// TODO Auto-generated method stub
		int n_samples = x.length;
		HashSet<Node> nodeSet = new HashSet<Node>();
		for (int i = 0; i < n_samples; i++) {
			Node leaf = tree.apply(x[i]);
			leaf.diff.add(y[i] - y_pred[i]);
			leaf.diff_sample_weight.add(sample_weight[i]);
			nodeSet.add(leaf);
		}
		for (Node leaf : nodeSet) {
			leaf.setTreeValue(Utils.averavge(leaf.diff, leaf.diff_sample_weight));
			// System.out.println(leaf+" "+leaf.treeVal);
			leaf.clear();
		}
	}

}
