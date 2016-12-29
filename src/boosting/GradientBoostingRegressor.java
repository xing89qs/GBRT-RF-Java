package boosting;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;
import java.util.TreeSet;

import data.LabeledSample;
import decision_tree.DecisionRegressionTree;
import decision_tree.DecisionRegressionTreeLeafSpliter;
import decision_tree.DecisionTree;
import decision_tree.Node;
import objective.Estimator;
import objective.LossFunction;
import objective.QuantileEstimator;
import objective.QuantileLossFunction;
import objective.SquaresEstimator;
import objective.SquaresLossFunction;
import util.BoostingListener;
import util.ParamException;
import util.ParamReader;
import util.dump.TreeInfo;

public class GradientBoostingRegressor {

	// ------------------------------------------------------------------------------------
	// Boosting的参数
	private double learning_rate;
	public int n_estimator;
	private double alpha; // quantile loss才需要
	private double sample_rate;
	private double feature_rate;
	private Estimator init_; // 初始值估计
	private LossFunction loss; // lossFunction
	private int random_state;
	private DecisionTree[] trees;
	private int max_depth;
	private int num_leafs; // 基于叶子节点分裂时的叶子数
	private double baseValue; // 初始值
	private double[] residual; // 残差
	private String spliter;
	private double min_leaf_sample;

	public GradientBoostingRegressor(HashMap<String, Object> params) throws ParamException {

		// -----------------------------------------------------------------------
		// 必需参数

		assert (params.containsKey("learning_rate"));
		this.learning_rate = ParamReader.readDouble("learning_rate", params);
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
		// 可选参数

		String objective = ParamReader.readString("objective", params);
		if (objective.equals("quantile")) {
			assert (params.containsKey("alpha"));
			this.alpha = ParamReader.readDouble("alpha", params);
			this.loss = new QuantileLossFunction(alpha);
			this.init_ = new QuantileEstimator(alpha);
		} else if (objective.equals("lad")) {
		} else {
			this.loss = new SquaresLossFunction();
			this.init_ = new SquaresEstimator();
		}

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

	private LabeledSample[] trainData;
	private LabeledSample[][] preSortedSampleArrays;
	private LabeledSample[][] copyOfPreSortedSampleArrays;
	private BoostingListener listener;

	private double[] _fit_stage(int i, double[][] X, double[] Y, double[] y_pred, double[] sample_weight) {
		this.loss.negative_gradient(Y, y_pred, this.residual);
		int featureNum = X[0].length;
		int sampleNum = X.length;
		for (int b = 0; b < sampleNum; b++)
			trainData[b].y = residual[b];
		for (int a = 0; a < featureNum; a++) {
			for (int b = 0; b < sampleNum; b++) {
				preSortedSampleArrays[a][b] = copyOfPreSortedSampleArrays[a][b];
			}
		}
		if (this.spliter.equals("leaf"))
			trees[i] = new DecisionRegressionTreeLeafSpliter(this.num_leafs, this.random_state, sample_rate,
					feature_rate, min_leaf_sample, preSortedSampleArrays);
		else
			trees[i] = new DecisionRegressionTree(i, this.max_depth, this.random_state, sample_rate, feature_rate,
					min_leaf_sample, preSortedSampleArrays);
		trees[i].fit(X, residual, sample_weight);

		this.loss.update_terminal_region(trees[i], X, Y, y_pred, sample_weight);
		for (int j = 0; j < X.length; j++) {
			Node leaf = trees[i].apply(X[j]);
			y_pred[j] += learning_rate * leaf.treeVal;
			// System.out.print(y_pred[j] + " ");
		}
		// System.out.println();
		return y_pred;
	}

	public void registerListener(BoostingListener listener) {
		this.listener = listener;
	}

	private int _fit_stages(double[][] X, double[] Y, double[] y_pred, double[] sample_weight, TreeInfo[][] infoList) {
		int featureNum = X[0].length;
		int sampleNum = X.length;
		trainData = new LabeledSample[sampleNum];
		for (int i = 0; i < sampleNum; i++) {
			trainData[i] = new LabeledSample();
			trainData[i].x = X[i];
			trainData[i].y = Y[i];
			trainData[i].weight = sample_weight[i];
		}

		preSortedSampleArrays = new LabeledSample[featureNum][sampleNum];
		copyOfPreSortedSampleArrays = new LabeledSample[featureNum][sampleNum];
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
		this.residual = new double[sampleNum];
		this.trees = new DecisionTree[this.n_estimator];

		for (int i = 0; i < this.n_estimator; i++) {
			System.out.println("第" + i + "棵树");
			if (listener != null)
				listener.done(i);
			y_pred = _fit_stage(i, X, Y, y_pred, sample_weight);
		}
		return this.n_estimator;
	}

	public void fit(double[][] X, double[] Y, double[] sample_weight, TreeInfo[][] infoList) {
		this.init_.fit(X, Y, sample_weight);
		double[] y_pred = this.init_.predict(X);
		this.baseValue = y_pred[0];
		_fit_stages(X, Y, y_pred, sample_weight, infoList);
	}

	public double[] predict(double[][] X) {
		double[] ans = new double[X.length];
		for (int i = 0; i < X.length; i++) {
			ans[i] = baseValue;
			for (int j = 0; j < n_estimator; j++) {
				Node leaf = trees[j].apply(X[i]);
				ans[i] += learning_rate * leaf.treeVal;
			}
		}
		return ans;
	}

	public double[] predict(double[][] X, int n) {
		double[] ans = new double[X.length];
		for (int i = 0; i < X.length; i++) {
			ans[i] = baseValue;
			for (int j = 0; j < n; j++) {
				Node leaf = trees[j].apply(X[i]);
				ans[i] += learning_rate * leaf.treeVal;
			}
		}
		return ans;
	}
}