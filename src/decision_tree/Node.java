package decision_tree;

import java.util.ArrayList;

public class Node {
	double split_val;
	int split_feature;
	public Node leftNode;
	public Node rightNode;
	public double treeVal;

	// 落在当前叶子节点的样本的统计信息
	public ArrayList<Double> diff = new ArrayList<Double>();
	public ArrayList<Double> diff_sample_weight = new ArrayList<Double>();

	public void setTreeValue(double value) {
		this.treeVal = value;
	}

	public void clear() {
		this.diff.clear();
		this.diff.trimToSize();
		this.diff_sample_weight.clear();
		this.diff_sample_weight.trimToSize();
		this.diff = null;
		this.diff_sample_weight = null;
	}
}