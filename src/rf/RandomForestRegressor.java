package rf;

import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;

import data.LabeledSample;
import decision_tree.DecisionRegressionTree;
import decision_tree.DecisionRegressionTreeLeafSpliter;
import decision_tree.DecisionTree;
import decision_tree.Node;
import objective.LossFunction;
import objective.SquaresLossFunction;
import util.ParamException;
import util.ParamReader;

public class RandomForestRegressor {

	private String spliter;
	private int num_leafs;
	private int max_depth;
	private int n_estimator;
	private int random_state;
	private double sample_rate;
	private double feature_rate;
	private double min_leaf_sample;
	private DecisionTree[] tree;
	private LabeledSample[] trainData;
	private LabeledSample[][] preSortedSampleArrays;
	private LabeledSample[][] copyOfPreSortedSampleArrays;
	private LossFunction loss;

	public RandomForestRegressor(HashMap<String, Object> params) throws ParamException {

		// -----------------------------------------------------------------------
		// 必需参数

		assert (params.containsKey("spliter"));
		this.spliter = ParamReader.readString("spliter", params);
		if (this.spliter.equals("leaf")) {
			assert (params.containsKey("num_leafs"));
			this.num_leafs = ParamReader.readInt("num_leafs", params);
		} else {
			assert (params.containsKey("max_depth"));
			this.max_depth = ParamReader.readInt("max_depth", params);
		}
		assert (params.containsKey("n_estimator"));
		this.n_estimator = ParamReader.readInt("n_estimator", params);

		// -------------------------------------------------------------------------
		if (params.containsKey("random_state"))
			this.random_state = ParamReader.readInt("random_state", params);
		else
			this.random_state = 0;
		if (params.containsKey("sample_rate"))
			this.sample_rate = ParamReader.readDouble("sample_rate", params);
		else
			this.sample_rate = 1.0;
		if (params.containsKey("feature_rate"))
			this.feature_rate = ParamReader.readDouble("feature_rate", params);
		else
			this.feature_rate = 1.0;

		if (params.containsKey("min_leaf_sample"))
			this.min_leaf_sample = ParamReader.readDouble("min_leaf_sample", params);
		else
			this.min_leaf_sample = 1;
	}

	public void fit(double[][] X, double[] Y, double[] sample_weight) {

		int featureNum = X[0].length;
		int sampleNum = X.length;
		this.trainData = new LabeledSample[sampleNum];
		for (int i = 0; i < sampleNum; i++) {
			trainData[i] = new LabeledSample();
			trainData[i].x = X[i];
			trainData[i].y = Y[i];
			trainData[i].weight = sample_weight[i];
		}
		this.loss = new SquaresLossFunction();

		this.preSortedSampleArrays = new LabeledSample[featureNum][sampleNum];
		this.copyOfPreSortedSampleArrays = new LabeledSample[featureNum][sampleNum];
		for (int a = 0; a < featureNum; a++) {
			for (int b = 0; b < sampleNum; b++) {
				copyOfPreSortedSampleArrays[a][b] = trainData[b];
			}

			final int compareFeature = a;
			Arrays.sort(copyOfPreSortedSampleArrays[a], 0, sampleNum, new Comparator<LabeledSample>() {

				@Override
				public int compare(LabeledSample o1, LabeledSample o2) {
					return new Double(o1.x[compareFeature]).compareTo(o2.x[compareFeature]);
				}
			});
		}

		this.tree = new DecisionTree[this.n_estimator];
		for (int i = 0; i < n_estimator; i++) {
			for (int a = 0; a < featureNum; a++) {
				for (int b = 0; b < sampleNum; b++) {
					preSortedSampleArrays[a][b] = copyOfPreSortedSampleArrays[a][b];
				}
			}
			if (this.spliter.equals("leaf"))
				this.tree[i] = new DecisionRegressionTreeLeafSpliter(this.num_leafs, i, sample_rate, feature_rate,
						min_leaf_sample, preSortedSampleArrays);
			else
				this.tree[i] = new DecisionRegressionTree(i, this.max_depth, i, sample_rate, feature_rate,
						min_leaf_sample, preSortedSampleArrays);
			System.out.println("第" + i + "棵树");
			this.tree[i].fit(X, Y, sample_weight);
			double[] pred = new double[X.length];
			this.loss.update_terminal_region(this.tree[i], X, Y, pred, sample_weight);
		}
	}

	public double[][] predict(double[][] X) {
		double ans[][] = new double[this.n_estimator][X.length];
		for (int i = 0; i < this.n_estimator; i++) {
			for (int j = 0; j < X.length; j++) {
				ans[i][j] = this.tree[i].apply(X[j]).treeVal;
			}
		}
		return ans;
	}
}
