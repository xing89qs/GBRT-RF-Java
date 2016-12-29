package decision_tree;

import java.util.ArrayList;
import java.util.Random;
import java.util.Stack;

import data.LabeledSample;

public class DecisionRegressionTree extends DecisionTree {

	// 树的参数
	private int max_depth;
	private int random_state;
	private Random random;
	private Node root;
	private double sample_rate;
	private double feature_rate;
	private int estimator_num;
	private LabeledSample[][] sampleSortedByFeatureArrays;
	private double min_leaf_sample;

	public DecisionRegressionTree(int estimator_num, int max_depth, int random_state, double sample_rate,
			double feature_rate, double min_leaf_sample, LabeledSample[][] preSortedSampleArrays) {
		this.estimator_num = estimator_num;
		this.max_depth = max_depth;
		this.random_state = random_state;
		System.out.println(this.random_state);
		this.random = new Random(this.random_state);
		this.sample_rate = sample_rate;
		this.feature_rate = feature_rate;
		this.min_leaf_sample = min_leaf_sample;
		this.sampleSortedByFeatureArrays = preSortedSampleArrays;
	}

	// public DecisionRegressionTree(int estimator_num, TreeInfo[] infoList) {
	// this.estimator_num = estimator_num;
	// Node[] nodeList = new Node[infoList.length];
	// for (int i = 0; i < infoList.length; i++) {
	// if (infoList[i] == null)
	// continue;
	// Node node = new Node();
	// nodeList[(int) infoList[i].root_id] = node;
	// if ((int) infoList[i].left_son != -1)
	// node.leftNode = nodeList[(int) infoList[i].left_son];
	// if ((int) infoList[i].right_son != -1)
	// node.rightNode = nodeList[(int) infoList[i].right_son];
	// node.split_feature = (int) infoList[i].split_feature;
	// node.split_val = infoList[i].split_feature_value;
	// node.treeVal = infoList[i].node_value;
	// if (infoList[i].is_root == 1L)
	// this.root = node;
	// }
	// }

	public DecisionRegressionTree() {
		// TODO Auto-generated constructor stub
	}

	class SplitResult {
		Stack<LabeledSample> leftSample, rightSample;
		double split_error, left_error, right_error;
		int best_feature;
		double best_split_val;

		// error = var_left + var_right
		// var_left = sigma((y_i-y_bar)^2*w)
		// = sigma(y_i*y_i*w_i)- y_bar*sigma(2*w_i*y_i)+ y_bar*y_bar*sigma(w_i)
		// y_bar = sigma(y_i*w_i)/sigma(w_i)

		double left_yyw_sum, left_y_w_sum, left_w_sum, left_bar;
		double right_yyw_sum, right_y_w_sum, right_w_sum, right_bar;

		public SplitResult(int l, int r, LabeledSample[] samples) {
			leftSample = new Stack<LabeledSample>();
			rightSample = new Stack<LabeledSample>();
			init(l, r, samples);
		}

		void init(int l, int r, LabeledSample[] samples) {
			this.split_error = Double.MAX_VALUE;
			leftSample.clear();
			rightSample.clear();

			left_yyw_sum = left_y_w_sum = left_w_sum = left_bar = 0;
			right_yyw_sum = right_y_w_sum = right_w_sum = right_bar = 0;
			for (int i = r; i >= l; i--) {
				if (!samples[i].isSampled)
					continue;
				rightSample.push(samples[i]);
				right_yyw_sum += samples[i].y * samples[i].y * samples[i].weight;
				right_y_w_sum += samples[i].y * samples[i].weight;
				right_w_sum += samples[i].weight;
			}
			right_bar = right_y_w_sum / right_w_sum;
			left_error = left_yyw_sum - 2 * left_y_w_sum * left_bar + left_bar * left_bar * left_w_sum;
			right_error = right_yyw_sum - 2 * right_y_w_sum * right_bar + right_bar * right_bar * right_w_sum;
			split_error = left_error + right_error;
		}

		void moveSampleToLeft(int split_feature, double split_val) {
			while (!rightSample.empty()) {
				LabeledSample sample = rightSample.peek();
				if (sample.x[split_feature] < split_val) {
					rightSample.pop();
					right_yyw_sum -= sample.y * sample.y * sample.weight;
					right_y_w_sum -= sample.y * sample.weight;
					right_w_sum -= sample.weight;

					leftSample.push(sample);
					left_yyw_sum += sample.y * sample.y * sample.weight;
					left_y_w_sum += sample.y * sample.weight;
					left_w_sum += sample.weight;
				} else
					break;
			}
			left_bar = left_y_w_sum / left_w_sum;
			right_bar = right_y_w_sum / right_w_sum;
			left_error = left_yyw_sum - 2 * left_y_w_sum * left_bar + left_bar * left_bar * left_w_sum;
			right_error = right_yyw_sum - 2 * right_y_w_sum * right_bar + right_bar * right_bar * right_w_sum;
			split_error = left_error + right_error;
		}

		Stack<LabeledSample> getLeftTrainSamples() {
			return leftSample;
		}

		Stack<LabeledSample> getRightTrainSamples() {
			return rightSample;
		}

		void clear() {
			this.leftSample.clear();
			this.rightSample.clear();
			this.leftSample = this.rightSample = null;
		}

	}

	private static double[] splitValueArray = new double[400005];
	private static LabeledSample[] samples = new LabeledSample[400005];

	private SplitResult getBestSplit(int l, int r) {

		int featureNum = this.sampleSortedByFeatureArrays[0][0].x.length;
		double min_split_error = Double.MAX_VALUE;
		int best_feature = -1;
		double best_split_val = Double.MAX_VALUE;

		for (int i = l; i <= r; i++) {
			if (random.nextDouble() > sample_rate)
				this.sampleSortedByFeatureArrays[0][i].isSampled = false;
			else
				this.sampleSortedByFeatureArrays[0][i].isSampled = true;
		}
		SplitResult currentResult = new SplitResult(l, r, sampleSortedByFeatureArrays[0]);
		for (int i = 0; i < featureNum; i++) {
			if (random.nextDouble() > feature_rate)
				continue;
			int cnt = 0;
			for (int j = l; j < r; j++) {
				splitValueArray[cnt++] = (sampleSortedByFeatureArrays[i][j].x[i]
						+ sampleSortedByFeatureArrays[i][j + 1].x[i]) / 2.0;
			}
			if (cnt == 0)
				continue;
			currentResult.init(l, r, sampleSortedByFeatureArrays[i]);
			for (int j = 0; j < cnt; j++) {
				double split_val = splitValueArray[j];
				currentResult.moveSampleToLeft(i, split_val);
				if (currentResult.split_error < min_split_error
						&& currentResult.getLeftTrainSamples().size() >= min_leaf_sample
						&& currentResult.getRightTrainSamples().size() >= min_leaf_sample) {
					min_split_error = currentResult.split_error;
					best_feature = i;
					best_split_val = split_val;
				}
			}
		}
		if (best_feature == -1)
			return null;
		for (int i = l; i <= r; i++)
			sampleSortedByFeatureArrays[best_feature][i].isSampled = true;
		SplitResult result = new SplitResult(l, r, sampleSortedByFeatureArrays[best_feature]);
		result.moveSampleToLeft(best_feature, best_split_val);
		result.best_feature = best_feature;
		result.best_split_val = best_split_val;
		return result;

	}

	private Node createTree(int l, int r, int depth) {
		if (depth > max_depth)
			return null;
		Node root = new Node();
		if (l >= r)
			return root;
		SplitResult result = getBestSplit(l, r);
		if (result == null)
			return root;
		root.split_feature = result.best_feature;
		root.split_val = result.best_split_val;
		Stack<LabeledSample> leftSamples = result.getLeftTrainSamples();
		Stack<LabeledSample> rightSamples = result.getRightTrainSamples();
		for (LabeledSample sample : leftSamples)
			sample.isSplitToLeft = true;
		for (LabeledSample sample : rightSamples)
			sample.isSplitToLeft = false;
		int leftSize = leftSamples.size();
		int n_features = sampleSortedByFeatureArrays[0][0].x.length;
		for (int i = 0; i < n_features; i++) {
			int tot = l;
			for (int j = l; j <= r; j++) {
				if (sampleSortedByFeatureArrays[i][j].isSplitToLeft) {
					samples[tot++] = sampleSortedByFeatureArrays[i][j];
				}
			}
			for (int j = l; j <= r; j++) {
				if (!sampleSortedByFeatureArrays[i][j].isSplitToLeft)
					samples[tot++] = sampleSortedByFeatureArrays[i][j];
			}
			for (int j = l; j <= r; j++)
				sampleSortedByFeatureArrays[i][j] = samples[j];

		}
		result.clear();
		root.leftNode = createTree(l, l + leftSize - 1, depth + 1);
		root.rightNode = createTree(l + leftSize, r, depth + 1);
		return root;
	}

	public void fit(double[][] X, double[] Y, double[] sample_weight) {
		random_state = random.nextInt();
		this.root = createTree(0, X.length - 1, 0);
	}

	private Node dfs(Node root, double[] x, int depth) {
		if (root.leftNode == null)
			return root;
		if (x[root.split_feature] < root.split_val)
			return dfs(root.leftNode, x, depth + 1);
		return dfs(root.rightNode, x, depth + 1);
	}

	public Node apply(double[] x) {
		return dfs(root, x, 0);
	}

	// private int nodeCount = 0;

	// private int dfsNode(Node node, ArrayList<TreeInfo> infoList, int is_root)
	// {
	// if (node == null)
	// return -1;
	// int left_son = dfsNode(node.leftNode, infoList, 0);
	// int right_son = dfsNode(node.rightNode, infoList, 0);
	// int id = nodeCount++;
	// TreeInfo info = new TreeInfo(id, left_son, right_son, node.split_feature,
	// node.split_val, estimator_num,
	// is_root, node.treeVal);
	// infoList.add(info);
	// return id;
	// }
	//
	// public ArrayList<TreeInfo> getTreeInfo() {
	// nodeCount = 0;
	// ArrayList<TreeInfo> infoList = new ArrayList<TreeInfo>();
	// dfsNode(root, infoList, 1);
	// return infoList;
	// }
}